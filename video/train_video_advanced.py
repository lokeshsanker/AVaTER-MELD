"""Train selectable MobileNet/VideoMAE/TimeSformer/Swin video models and export 512-D caches.
On RTX 2050 4GB, start with MobileNet; transformer backbones may need batch-size 1.
"""
import argparse,sys,os,random
from pathlib import Path
import numpy as np,pandas as pd,torch,torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import accuracy_score,f1_score,classification_report
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from config import EMOTIONS,LABEL2IDX,NUM_FRAMES
from video.model_advanced import AdvancedVideoEmotionModel,VIDEO_BACKBONES
import torchvision.transforms as T
NORM=T.Normalize(mean=[.485,.456,.406],std=[.229,.224,.225])
class FaceDataset(Dataset):
 def __init__(self,split):
  self.df=pd.read_csv(ROOT/'data'/'manifests'/f'{split}.csv'); self.cache=ROOT/'data'/'face_cache'/split
 def __len__(self): return len(self.df)
 def __getitem__(self,i):
  r=self.df.iloc[i]; arr=np.load(self.cache/(str(r.clip_id)+'.npy'))
  if len(arr)>NUM_FRAMES: arr=arr[np.linspace(0,len(arr)-1,NUM_FRAMES).astype(int)]
  if len(arr)<NUM_FRAMES:
   if len(arr)==0: raise ValueError(f'No cached faces for {r.clip_id}')
   arr=np.concatenate([arr,np.repeat(arr[-1:],NUM_FRAMES-len(arr),axis=0)],axis=0)
  frames=torch.from_numpy(arr).float().permute(0,3,1,2)/255.; frames=torch.stack([NORM(f) for f in frames])
  return frames,LABEL2IDX[str(r.label)],str(r.clip_id)
def collate(b): return torch.stack([x[0] for x in b]),torch.tensor([x[1] for x in b]),[x[2] for x in b]
def seed_all(s): random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s) if torch.cuda.is_available() else None
def pass_split(model,loader,device,train=False,optimizer=None,criterion=None,scaler=None,amp=False):
 model.train(train);losses=[];ys=[];ps=[];feats=[];ids=[]
 for frames,y,clip in loader:
  frames=frames.to(device);y=y.to(device)
  if train: optimizer.zero_grad(set_to_none=True)
  with torch.set_grad_enabled(train):
   with torch.autocast(device_type=device.type,enabled=amp): logits,emb=model(frames,return_embedding=True); loss=criterion(logits,y) if train else None
   if train: scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);scaler.step(optimizer);scaler.update();losses.append(loss.item())
  ys.extend(y.detach().cpu().tolist());ps.extend(logits.argmax(1).detach().cpu().tolist());feats.append(emb.detach().cpu());ids.extend(clip)
 return {'loss':float(np.mean(losses)) if losses else None,'accuracy':float(accuracy_score(ys,ps)),'macro_f1':float(f1_score(ys,ps,labels=list(range(7)),average='macro',zero_division=0)),'weighted_f1':float(f1_score(ys,ps,labels=list(range(7)),average='weighted',zero_division=0)),'y':ys,'pred':ps,'features':torch.cat(feats),'clip_ids':ids}
def main():
 p=argparse.ArgumentParser();p.add_argument('--backbone',choices=VIDEO_BACKBONES,default='mobilenet');p.add_argument('--epochs',type=int,default=12);p.add_argument('--batch-size',type=int,default=1);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--seed',type=int,default=42);p.add_argument('--unfreeze-backbone',action='store_true');a=p.parse_args()
 seed_all(a.seed);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');amp=device.type=='cuda';print('Device:',device,'backbone:',a.backbone,'AMP:',amp)
 sets={s:FaceDataset(s) for s in ('train','dev','test')}; loaders={s:DataLoader(d,batch_size=a.batch_size,shuffle=(s=='train'),num_workers=0,collate_fn=collate,pin_memory=amp) for s,d in sets.items()}
 model=AdvancedVideoEmotionModel(a.backbone,freeze_backbone=not a.unfreeze_backbone).to(device);criterion=nn.CrossEntropyLoss(label_smoothing=.1);optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.lr,weight_decay=1e-4);scaler=torch.amp.GradScaler('cuda',enabled=amp);sched=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=a.epochs)
 ckdir=ROOT/'checkpoints'/'video_advanced';ckdir.mkdir(parents=True,exist_ok=True);ckpt=ckdir/f'{a.backbone}_best.pt';best=-1;wait=0;history=[]
 for ep in range(1,a.epochs+1):
  tr=pass_split(model,loaders['train'],device,True,optimizer,criterion,scaler,amp);dv=pass_split(model,loaders['dev'],device);sched.step();history.append({'epoch':ep,'train_loss':tr['loss'],'train_macro_f1':tr['macro_f1'],'dev_accuracy':dv['accuracy'],'dev_macro_f1':dv['macro_f1'],'dev_weighted_f1':dv['weighted_f1']});print(f"Epoch {ep}/{a.epochs} train_loss={tr['loss']:.4f} train_macro={tr['macro_f1']:.4f} dev_acc={dv['accuracy']:.4f} dev_macro={dv['macro_f1']:.4f} dev_weighted={dv['weighted_f1']:.4f}")
  if dv['macro_f1']>best:best=dv['macro_f1'];wait=0;torch.save({'model_state_dict':model.state_dict(),'backbone':a.backbone,'best_dev_macro_f1':best},ckpt)
  else:wait+=1
  if wait>=4:print('Early stopping');break
 import pandas as pd;pd.DataFrame(history).to_csv(ROOT/'results'/f'video_advanced_{a.backbone}_history.csv',index=False);state=torch.load(ckpt,map_location=device,weights_only=False);model.load_state_dict(state['model_state_dict'])
 for split,loader in loaders.items():
  export_loader=DataLoader(sets[split],batch_size=a.batch_size,shuffle=False,num_workers=0,collate_fn=collate,pin_memory=amp)
  r=pass_split(model,export_loader,device);torch.save({'features':r['features'],'labels':r['y'],'clip_ids':r['clip_ids'],'feature_dim':512,'num_samples':len(r['y']),'model':f'{a.backbone}+temporal attention','manifest':f'{split}.csv'},ROOT/'data'/f'video_advanced_feature_cache_{split}.pt');print(split,'accuracy',r['accuracy'],'macro_f1',r['macro_f1'],'weighted_f1',r['weighted_f1'])
  if split=='test':
   (ROOT/'results'/f'video_advanced_{a.backbone}_test_report.txt').write_text(classification_report(r['y'],r['pred'],labels=list(range(7)),target_names=EMOTIONS,digits=4,zero_division=0),encoding='utf-8')
   import json
   (ROOT/'results'/f'video_advanced_{a.backbone}_test_metrics.json').write_text(json.dumps({k:r[k] for k in ('accuracy','macro_f1','weighted_f1')},indent=2),encoding='utf-8')
 print('Checkpoint:',ckpt);print('Feature caches: data/video_advanced_feature_cache_{train,dev,test}.pt')
if __name__=='__main__':main()
