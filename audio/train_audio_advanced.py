"""Train sequence-aware Wav2Vec2/HuBERT audio branch and export 768-D caches.
Requires the existing MELD manifests and audio files. Use small batches on 4-GB GPUs.
"""
import argparse, os, sys, random
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, librosa
from torch.utils.data import Dataset,DataLoader
from transformers import AutoFeatureExtractor
from sklearn.metrics import accuracy_score,f1_score,classification_report
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from config import EMOTIONS,LABEL2IDX,AUDIO_SAMPLE_RATE
from audio.model_advanced import AudioSequenceEmotionModel,BACKBONES

class WaveDataset(Dataset):
 def __init__(self,manifest,extractor,seconds=3.5): self.df=pd.read_csv(manifest); self.n=int(seconds*AUDIO_SAMPLE_RATE); self.extractor=extractor
 def __len__(self): return len(self.df)
 def __getitem__(self,i):
  r=self.df.iloc[i]; wav,_=librosa.load(r.wav_path,sr=AUDIO_SAMPLE_RATE,mono=True)
  wav=wav[:self.n] if len(wav)>self.n else np.pad(wav,(0,self.n-len(wav)))
  normalized=self.extractor(wav,sampling_rate=AUDIO_SAMPLE_RATE,return_tensors='np')['input_values'][0]
  return torch.tensor(normalized,dtype=torch.float32),LABEL2IDX[str(r.label)],str(r.clip_id)

def collate(batch): return torch.stack([x[0] for x in batch]),torch.tensor([x[1] for x in batch]),[x[2] for x in batch]
def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s) if torch.cuda.is_available() else None

def collect(model,loader,device,train=False,optimizer=None,criterion=None,scaler=None,amp=False,accumulation_steps=1):
 model.train(train); losses=[]; ys=[]; ps=[]; feats=[]; ids=[]
 for step,(wav,y,clip) in enumerate(loader):
  wav=wav.to(device); y=y.to(device)
  if train and step % accumulation_steps == 0: optimizer.zero_grad(set_to_none=True)
  with torch.set_grad_enabled(train):
   with torch.autocast(device_type=device.type,enabled=amp): logits,emb=model(wav,return_embedding=True); loss=criterion(logits,y) if train else None
   if train:
    scaler.scale(loss / accumulation_steps).backward();
    if (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader):
     scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(optimizer); scaler.update()
    losses.append(loss.item())
  ys.extend(y.detach().cpu().tolist()); ps.extend(logits.argmax(1).detach().cpu().tolist()); feats.append(emb.detach().cpu()); ids.extend(clip)
 return {'loss':float(np.mean(losses)) if losses else None,'accuracy':float(accuracy_score(ys,ps)),'macro_f1':float(f1_score(ys,ps,labels=list(range(7)),average='macro',zero_division=0)),'weighted_f1':float(f1_score(ys,ps,labels=list(range(7)),average='weighted',zero_division=0)),'y':ys,'pred':ps,'features':torch.cat(feats),'clip_ids':ids}

def main():
 p=argparse.ArgumentParser(); p.add_argument('--backbone',choices=BACKBONES,default='wav2vec2'); p.add_argument('--epochs',type=int,default=12); p.add_argument('--batch-size',type=int,default=2); p.add_argument('--accumulation-steps',type=int,default=4); p.add_argument('--lr',type=float,default=1e-4); p.add_argument('--seed',type=int,default=42); p.add_argument('--unfreeze-encoder',action='store_true'); a=p.parse_args()
 seed_all(a.seed); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); amp=device.type=='cuda'; print('Device:',device,'AMP:',amp,'backbone:',a.backbone)
 extractor=AutoFeatureExtractor.from_pretrained(BACKBONES[a.backbone])
 data={s:WaveDataset(ROOT/'data'/'manifests'/f'{s}.csv',extractor) for s in ('train','dev','test')}
 loaders={s:DataLoader(d,batch_size=a.batch_size,shuffle=(s=='train'),num_workers=0,collate_fn=collate,pin_memory=amp) for s,d in data.items()}
 model=AudioSequenceEmotionModel(a.backbone,freeze_encoder=not a.unfreeze_encoder).to(device); criterion=nn.CrossEntropyLoss(label_smoothing=.1); optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.lr,weight_decay=1e-4); scaler=torch.amp.GradScaler('cuda',enabled=amp); sched=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=a.epochs)
 out=ROOT/'checkpoints'/'audio_advanced'; out.mkdir(parents=True,exist_ok=True); ckpt=out/f'{a.backbone}_best.pt'; best=-1; wait=0; hist=[]
 for ep in range(1,a.epochs+1):
  tr=collect(model,loaders['train'],device,True,optimizer,criterion,scaler,amp,a.accumulation_steps); dv=collect(model,loaders['dev'],device); sched.step(); hist.append({'epoch':ep,'train_loss':tr['loss'],'train_macro_f1':tr['macro_f1'],'dev_accuracy':dv['accuracy'],'dev_macro_f1':dv['macro_f1'],'dev_weighted_f1':dv['weighted_f1']}); print(f"Epoch {ep}/{a.epochs} train_loss={tr['loss']:.4f} train_macro={tr['macro_f1']:.4f} dev_acc={dv['accuracy']:.4f} dev_macro={dv['macro_f1']:.4f} dev_weighted={dv['weighted_f1']:.4f}")
  if dv['macro_f1']>best: best=dv['macro_f1']; wait=0; torch.save({'model_state_dict':model.state_dict(),'backbone':a.backbone,'best_dev_macro_f1':best},ckpt)
  else: wait+=1
  if wait>=4: print('Early stopping'); break
 import pandas as pd; pd.DataFrame(hist).to_csv(ROOT/'results'/f'audio_advanced_{a.backbone}_history.csv',index=False)
 state=torch.load(ckpt,map_location=device,weights_only=False); model.load_state_dict(state['model_state_dict'])
 for split,loader in loaders.items():
  export_loader=DataLoader(data[split],batch_size=a.batch_size,shuffle=False,num_workers=0,collate_fn=collate,pin_memory=amp)
  result=collect(model,export_loader,device); cache={'features':result['features'],'labels':result['y'],'clip_ids':result['clip_ids'],'feature_dim':768,'num_samples':len(result['y']),'model':f'{a.backbone}+BiLSTM+Attention','manifest':f'{split}.csv'}; torch.save(cache,ROOT/'data'/f'audio_advanced_feature_cache_{split}.pt'); print(split, 'accuracy',result['accuracy'],'macro_f1',result['macro_f1'],'weighted_f1',result['weighted_f1']);
  if split=='test':
   (ROOT/'results'/f'audio_advanced_{a.backbone}_test_report.txt').write_text(classification_report(result['y'],result['pred'],labels=list(range(7)),target_names=EMOTIONS,digits=4,zero_division=0),encoding='utf-8')
   import json
   (ROOT/'results'/f'audio_advanced_{a.backbone}_test_metrics.json').write_text(json.dumps({k:result[k] for k in ('accuracy','macro_f1','weighted_f1')},indent=2),encoding='utf-8')
 print('Checkpoint:',ckpt); print('Feature caches: data/audio_advanced_feature_cache_{train,dev,test}.pt')
if __name__=='__main__': main()
