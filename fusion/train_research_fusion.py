"""Research MER trainer for cached MELD modality features.

Supports loss_type={cross_entropy,weighted_ce,focal},
fusion_type={concat,gated,cross_attention}, modality ablations,
Macro-F1 early stopping, calibration, full metrics, plots and error analysis.

Default inputs are the existing improved caches (Text V2, Audio V4, Video V3).
This script never overwrites V6/V7 checkpoints.
"""
from __future__ import annotations
import argparse, json, os, random, sys, math, csv
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from config import EMOTIONS

DEFAULT_CACHES = {
 'text': {'train':'data/text_v2_feature_cache_train.pt','dev':'data/text_v2_feature_cache_dev.pt','test':'data/text_v2_feature_cache_test.pt'},
 'audio': {'train':'data/audio_v4_feature_cache_train.pt','dev':'data/audio_v4_feature_cache_dev.pt','test':'data/audio_v4_feature_cache_test.pt'},
 'video': {'train':'data/video_v3_feature_cache_train.pt','dev':'data/video_v3_feature_cache_dev.pt','test':'data/video_v3_feature_cache_test.pt'},
}
DIMS = {'text':768, 'audio':768, 'video':512}
LABEL2IDX = {name:i for i,name in enumerate(EMOTIONS)}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def labels_to_tensor(values):
    if torch.is_tensor(values):
        if values.dtype.is_floating_point: return values.long()
        if values.dtype in (torch.int8,torch.int16,torch.int32,torch.int64,torch.uint8): return values.long()
        values = values.tolist()
    mapped=[]
    for v in values:
        if isinstance(v, (int, np.integer)): mapped.append(int(v))
        elif str(v) in LABEL2IDX: mapped.append(LABEL2IDX[str(v)])
        else: raise ValueError(f'Unknown label in feature cache: {v!r}')
    return torch.tensor(mapped,dtype=torch.long)


def load_cache(path, modality):
    if not path.exists(): raise FileNotFoundError(f'Missing {modality} cache: {path}')
    obj=torch.load(path,map_location='cpu',weights_only=False)
    if torch.is_tensor(obj): return {'features':obj.float(),'labels':None,'clip_ids':None}
    if not isinstance(obj,dict) or 'features' not in obj: raise ValueError(f'Unsupported cache format: {path}')
    labels=obj.get('labels')
    return {'features':obj['features'].float(), 'labels':labels_to_tensor(labels) if labels is not None else None,
            'clip_ids':obj.get('clip_ids')}


def load_splits(cache_overrides):
    loaded={}
    for split in ('train','dev','test'):
        loaded[split]={}
        for modality in ('text','audio','video'):
            path=Path(cache_overrides.get(modality,{}).get(split, DEFAULT_CACHES[modality][split]))
            if not path.is_absolute(): path=ROOT/path
            data=load_cache(path,modality)
            x=data['features']
            if x.ndim != 2: raise ValueError(f'{modality}/{split} expected [N,D], got {tuple(x.shape)}')
            if x.shape[1] != DIMS[modality]: raise ValueError(f'{modality}/{split} expected dim {DIMS[modality]}, got {x.shape[1]} at {path}')
            loaded[split][modality]=data
            print(f'{split:5s} {modality:5s}: {tuple(x.shape)} ({path})')
        txt=loaded[split]['text']; aud=loaded[split]['audio']; vid=loaded[split]['video']
        sizes=[len(txt['features']),len(aud['features']),len(vid['features'])]
        if len(set(sizes))!=1: raise ValueError(f'{split}: modality sample counts differ: {sizes}')
        if txt['labels'] is None: raise ValueError(f'Text cache must provide labels for {split}')
        for m in ('audio','video'):
            if loaded[split][m]['labels'] is not None and not torch.equal(txt['labels'],loaded[split][m]['labels']):
                raise ValueError(f'{split}: text/{m} label alignment mismatch')
        for m in ('audio','video'):
            ids=loaded[split][m]['clip_ids']
            if ids is not None and txt['clip_ids'] is not None and list(ids)!=list(txt['clip_ids']):
                raise ValueError(f'{split}: text/{m} clip ID alignment mismatch')
        print(f'{split} alignment: PASS; samples={sizes[0]}')
    return loaded


def class_weights(labels, device):
    counts=torch.bincount(labels,minlength=len(EMOTIONS)).float().clamp_min(1)
    # Inverse-square-root weights reduce imbalance without the extreme variance
    # of raw inverse-frequency weights on MELD's rare classes.
    weights=counts.rsqrt(); weights=weights/weights.mean()
    return weights.to(device), counts.cpu().tolist()


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2.0, label_smoothing=0.0):
        super().__init__(); self.register_buffer('alpha',alpha); self.gamma=gamma; self.smoothing=label_smoothing
    def forward(self, logits, target):
        logp=F.log_softmax(logits,dim=-1); p=logp.exp()
        nll=-logp.gather(1,target[:,None]).squeeze(1)
        smooth=-logp.mean(dim=-1)
        ce=(1-self.smoothing)*nll+self.smoothing*smooth
        pt=p.gather(1,target[:,None]).squeeze(1)
        alpha=self.alpha.gather(0,target)
        return (alpha*((1-pt).clamp_min(0)**self.gamma)*ce).mean()


def build_loss(kind, weights, smoothing, device):
    if kind=='cross_entropy': return nn.CrossEntropyLoss(label_smoothing=smoothing)
    if kind=='weighted_ce': return nn.CrossEntropyLoss(weight=weights,label_smoothing=smoothing)
    if kind=='focal': return FocalLoss(weights,gamma=2.0,label_smoothing=smoothing).to(device)
    raise ValueError(kind)


class ConcatFusion(nn.Module):
    def __init__(self, modalities, hidden=512, dropout=.3):
        super().__init__(); in_dim=sum(DIMS[m] for m in modalities)
        self.net=nn.Sequential(nn.LayerNorm(in_dim),nn.Linear(in_dim,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,len(EMOTIONS)))
    def forward(self, xs): return self.net(torch.cat(xs,dim=1)), None

class GatedFusion(nn.Module):
    def __init__(self, modalities, hidden=256, dropout=.2):
        super().__init__(); self.modalities=modalities
        self.proj=nn.ModuleDict({m:nn.Sequential(nn.LayerNorm(DIMS[m]),nn.Linear(DIMS[m],hidden),nn.GELU(),nn.Dropout(dropout)) for m in modalities})
        self.gate=nn.Sequential(nn.Linear(hidden*len(modalities),hidden),nn.GELU(),nn.Linear(hidden,len(modalities)))
        self.head=nn.Sequential(nn.LayerNorm(hidden),nn.Linear(hidden,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,len(EMOTIONS)))
    def forward(self,xs):
        h=[self.proj[m](x) for m,x in zip(self.modalities,xs)]
        gates=torch.softmax(self.gate(torch.cat(h,dim=1)),dim=1)
        fused=sum(gates[:,i:i+1]*v for i,v in enumerate(h))
        return self.head(fused),gates

class CrossAttentionFusion(nn.Module):
    def __init__(self, modalities, hidden=128, tokens=6, heads=4, dropout=.1):
        super().__init__(); self.modalities=modalities; self.tokens=tokens; self.hidden=hidden
        self.proj=nn.ModuleDict({m:nn.Sequential(nn.LayerNorm(DIMS[m]),nn.Linear(DIMS[m],tokens*hidden)) for m in modalities})
        self.cross=nn.MultiheadAttention(hidden,heads,dropout=dropout,batch_first=True)
        self.norm1=nn.LayerNorm(hidden); self.norm2=nn.LayerNorm(hidden)
        enc=nn.TransformerEncoderLayer(d_model=hidden,nhead=heads,dim_feedforward=hidden*4,dropout=dropout,activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(enc,num_layers=2)
        self.head=nn.Sequential(nn.Linear(hidden,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,len(EMOTIONS)))
    def forward(self,xs):
        tok={m:self.proj[m](x).view(x.shape[0],self.tokens,self.hidden) for m,x in zip(self.modalities,xs)}
        # Text is query when available; otherwise use first available modality.
        query_name='text' if 'text' in self.modalities else self.modalities[0]
        q=tok[query_name]
        context=[tok[m] for m in self.modalities if m!=query_name]
        kv=torch.cat(context,dim=1) if context else q
        attended,_=self.cross(q,kv,kv,need_weights=False)
        q=self.norm1(q+attended); q=self.norm2(q+self.encoder(q))
        pooled=q.mean(dim=1)
        return self.head(pooled),None


def make_model(fusion, modalities):
    if fusion=='concat': return ConcatFusion(modalities)
    if fusion=='gated': return GatedFusion(modalities)
    if fusion=='cross_attention': return CrossAttentionFusion(modalities)
    raise ValueError(fusion)


def metrics_from_logits(logits, labels):
    probs=torch.softmax(logits,dim=1).cpu().numpy(); pred=probs.argmax(axis=1); y=labels.cpu().numpy()
    return {'accuracy':float(accuracy_score(y,pred)), 'macro_f1':float(f1_score(y,pred,labels=list(range(len(EMOTIONS))),average='macro',zero_division=0)),
            'weighted_f1':float(f1_score(y,pred,labels=list(range(len(EMOTIONS))),average='weighted',zero_division=0)), 'y_true':y,'y_pred':pred,'probs':probs}

@torch.no_grad()
def infer(model, split_data, modalities, device, batch_size=128):
    model.eval(); xs=[split_data[m]['features'] for m in modalities]; y=split_data['text']['labels']
    ds=TensorDataset(*xs,y); loader=DataLoader(ds,batch_size=batch_size,shuffle=False)
    logits_all=[]; gates_all=[]
    for batch in loader:
        *xb,yb=batch; xb=[x.to(device) for x in xb]
        logits,gates=model(xb); logits_all.append(logits.cpu())
        if gates is not None: gates_all.append(gates.cpu())
    return torch.cat(logits_all), y, (torch.cat(gates_all) if gates_all else None)


def ece_score(probs, y, bins=15):
    conf=probs.max(axis=1); pred=probs.argmax(axis=1); correct=(pred==y).astype(float); ece=0.0
    edges=np.linspace(0,1,bins+1)
    rows=[]
    for i in range(bins):
        mask=(conf>edges[i]) & (conf<=edges[i+1] if i==bins-1 else conf<edges[i+1])
        if mask.any():
            acc=float(correct[mask].mean()); avg=float(conf[mask].mean()); frac=float(mask.mean())
            ece += frac*abs(acc-avg); rows.append({'bin_lower':edges[i],'bin_upper':edges[i+1],'count':int(mask.sum()),'confidence':avg,'accuracy':acc})
        else: rows.append({'bin_lower':edges[i],'bin_upper':edges[i+1],'count':0,'confidence':np.nan,'accuracy':np.nan})
    return float(ece),rows


def fit_temperature(dev_logits, dev_y, device):
    # Scalar temperature is fitted on validation logits only, never on test labels.
    logits=dev_logits.to(device); labels=dev_y.to(device)
    log_t=torch.zeros(1,device=device,requires_grad=True)
    optimizer=torch.optim.LBFGS([log_t],lr=.05,max_iter=80,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad(); t=log_t.exp().clamp(.05,20); loss=F.cross_entropy(logits/t,labels); loss.backward(); return loss
    optimizer.step(closure)
    return float(log_t.detach().exp().clamp(.05,20).cpu().item())


def save_eval(outdir, tag, logits, labels, clip_ids, temperature=1.0):
    outdir.mkdir(parents=True,exist_ok=True)
    probs=torch.softmax(logits/temperature,dim=1).numpy(); y=labels.numpy(); pred=probs.argmax(axis=1); conf=probs.max(axis=1); correct=pred==y
    report=classification_report(y,pred,labels=list(range(len(EMOTIONS))),target_names=EMOTIONS,output_dict=True,zero_division=0)
    (outdir/f'{tag}_classification_report.txt').write_text(classification_report(y,pred,labels=list(range(len(EMOTIONS))),target_names=EMOTIONS,digits=4,zero_division=0),encoding='utf-8')
    pd_rows=[]
    for e in EMOTIONS:
        r=report[e]; pd_rows.append({'emotion':e,'precision':r['precision'],'recall':r['recall'],'f1':r['f1-score'],'support':r['support']})
    import pandas as pd
    pd.DataFrame(pd_rows).to_csv(outdir/f'{tag}_per_class_metrics.csv',index=False)
    cm=confusion_matrix(y,pred,labels=list(range(len(EMOTIONS))))
    pd.DataFrame(cm,index=EMOTIONS,columns=EMOTIONS).to_csv(outdir/f'{tag}_confusion_matrix.csv')
    ece,reliability=ece_score(probs,y)
    pd.DataFrame(reliability).to_csv(outdir/f'{tag}_reliability_bins.csv',index=False)
    rows=[]
    ids=clip_ids if clip_ids is not None else [str(i) for i in range(len(y))]
    for i in range(len(y)):
        row={'clip_id':str(ids[i]),'actual':EMOTIONS[int(y[i])],'predicted':EMOTIONS[int(pred[i])], 'confidence':float(conf[i]),'confidence_percent':float(conf[i]*100),'correct':bool(correct[i]),'temperature':temperature}
        for j,e in enumerate(EMOTIONS): row[f'prob_{e}']=float(probs[i,j])
        rows.append(row)
    pd.DataFrame(rows).to_csv(outdir/f'{tag}_predictions.csv',index=False)
    pd.DataFrame([r for r in rows if not r['correct']]).sort_values('confidence',ascending=False).to_csv(outdir/f'{tag}_errors_high_confidence_first.csv',index=False)
    # Reliability diagram
    valid=[r for r in reliability if r['count']>0]
    plt.figure(figsize=(6,6)); plt.plot([0,1],[0,1],linestyle='--',label='Perfect calibration')
    if valid: plt.plot([r['confidence'] for r in valid],[r['accuracy'] for r in valid],marker='o',label='Model')
    plt.xlabel('Mean confidence'); plt.ylabel('Empirical accuracy'); plt.title(f'{tag}: reliability diagram (ECE={ece:.4f})'); plt.legend(); plt.tight_layout(); plt.savefig(outdir/f'{tag}_reliability.png',dpi=220); plt.close()
    plt.figure(figsize=(7,4)); plt.hist(conf,bins=20); plt.xlabel('Maximum class probability'); plt.ylabel('Samples'); plt.title(f'{tag}: confidence histogram'); plt.tight_layout(); plt.savefig(outdir/f'{tag}_confidence_histogram.png',dpi=220); plt.close()
    # ROC and PR curves per class (one-vs-rest)
    y_one=np.eye(len(EMOTIONS))[y]
    plt.figure(figsize=(7,6)); roc_rows=[]
    for i,e in enumerate(EMOTIONS):
        try:
            fpr,tpr,_=roc_curve(y_one[:,i],probs[:,i]); auc=float(roc_auc_score(y_one[:,i],probs[:,i])); plt.plot(fpr,tpr,label=f'{e} AUC={auc:.3f}'); roc_rows.append({'emotion':e,'roc_auc':auc})
        except ValueError: roc_rows.append({'emotion':e,'roc_auc':None})
    plt.plot([0,1],[0,1],'k--'); plt.xlabel('False positive rate'); plt.ylabel('True positive rate'); plt.title(f'{tag}: one-vs-rest ROC'); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(outdir/f'{tag}_roc_curves.png',dpi=220); plt.close()
    plt.figure(figsize=(7,6)); pr_rows=[]
    for i,e in enumerate(EMOTIONS):
        try:
            precision,recall,_=precision_recall_curve(y_one[:,i],probs[:,i]); ap=float(average_precision_score(y_one[:,i],probs[:,i])); plt.plot(recall,precision,label=f'{e} AP={ap:.3f}'); pr_rows.append({'emotion':e,'average_precision':ap})
        except ValueError: pr_rows.append({'emotion':e,'average_precision':None})
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'{tag}: one-vs-rest PR curves'); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(outdir/f'{tag}_pr_curves.png',dpi=220); plt.close()
    pd.DataFrame(roc_rows).to_csv(outdir/f'{tag}_roc_auc.csv',index=False); pd.DataFrame(pr_rows).to_csv(outdir/f'{tag}_average_precision.csv',index=False)
    return {'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,labels=list(range(len(EMOTIONS))),average='macro',zero_division=0)),
            'weighted_f1':float(f1_score(y,pred,labels=list(range(len(EMOTIONS))),average='weighted',zero_division=0)), 'ece':ece,'mean_confidence':float(conf.mean()),'mean_confidence_correct':float(conf[correct].mean()) if correct.any() else None,'mean_confidence_incorrect':float(conf[~correct].mean()) if (~correct).any() else None,'temperature':temperature}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--modalities',nargs='+',choices=['text','audio','video'],default=['text','audio','video'])
    ap.add_argument('--fusion-type',choices=['concat','gated','cross_attention'],default='concat')
    ap.add_argument('--loss-type',choices=['cross_entropy','weighted_ce','focal'],default='weighted_ce')
    ap.add_argument('--label-smoothing',type=float,default=.1)
    ap.add_argument('--epochs',type=int,default=30); ap.add_argument('--batch-size',type=int,default=64)
    ap.add_argument('--patience',type=int,default=6); ap.add_argument('--lr',type=float,default=3e-4); ap.add_argument('--weight-decay',type=float,default=1e-4)
    ap.add_argument('--seed',type=int,default=42); ap.add_argument('--run-name',default=None)
    ap.add_argument('--text-cache-dir',default='data'); ap.add_argument('--audio-cache-dir',default='data'); ap.add_argument('--video-cache-dir',default='data')
    ap.add_argument('--text-cache-prefix',default='text_v2_feature_cache'); ap.add_argument('--audio-cache-prefix',default='audio_v4_feature_cache'); ap.add_argument('--video-cache-prefix',default='video_v3_feature_cache')
    ap.add_argument('--no-amp',action='store_true'); ap.add_argument('--tensorboard',action='store_true'); ap.add_argument('--wandb',action='store_true'); ap.add_argument('--wandb-project',default='meld-mer-research')
    args=ap.parse_args(); set_seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); amp=(device.type=='cuda' and not args.no_amp)
    overrides={
        'text': {s:str(Path(args.text_cache_dir)/(args.text_cache_prefix+f'_{s}.pt')) for s in ('train','dev','test')},
        'audio': {s:str(Path(args.audio_cache_dir)/(args.audio_cache_prefix+f'_{s}.pt')) for s in ('train','dev','test')},
        'video': {s:str(Path(args.video_cache_dir)/(args.video_cache_prefix+f'_{s}.pt')) for s in ('train','dev','test')},
    }
    splits=load_splits(overrides)
    mods=args.modalities
    run_name=args.run_name or f'{args.fusion_type}_{args.loss_type}_'+'_'.join(mods)
    outdir=ROOT/'results'/'research_runs'/run_name; ckptdir=ROOT/'checkpoints'/'research_runs'; ckptdir.mkdir(parents=True,exist_ok=True); outdir.mkdir(parents=True,exist_ok=True)
    writer=None; wandb_run=None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as e:
            raise RuntimeError('TensorBoard requested but not installed. Run: pip install tensorboard') from e
        writer=SummaryWriter(log_dir=str(outdir/'tensorboard'))
    if args.wandb:
        try: import wandb
        except ImportError as e: raise RuntimeError('W&B requested but not installed. Run: pip install wandb') from e
        wandb_run=wandb.init(project=args.wandb_project,name=run_name,config=vars(args))
    train=splits['train']; dev=splits['dev']; test=splits['test']; ytrain=train['text']['labels']; ydev=dev['text']['labels']; ytest=test['text']['labels']
    w,counts=class_weights(ytrain,device); criterion=build_loss(args.loss_type,w,args.label_smoothing,device)
    train_ds=TensorDataset(*[train[m]['features'] for m in mods],ytrain); train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,pin_memory=amp)
    model=make_model(args.fusion_type,mods).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs,eta_min=1e-6)
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    best=-1.; wait=0; history=[]; checkpoint=ckptdir/f'{run_name}_best.pt'
    print(f'\nRun={run_name} device={device} fusion={args.fusion_type} loss={args.loss_type} modalities={mods} AMP={amp}')
    print('Training class counts:',dict(zip(EMOTIONS,counts))); print('Class weights:',w.detach().cpu().numpy().round(3).tolist())
    for epoch in range(1,args.epochs+1):
        model.train(); total=0.; n=0; ys=[]; ps=[]
        for batch in train_loader:
            *xb,yb=batch; xb=[x.to(device,non_blocking=True) for x in xb]; yb=yb.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=amp): logits,_=model(xb); loss=criterion(logits,yb)
            if not torch.isfinite(loss): raise RuntimeError(f'Non-finite loss at epoch {epoch}: {loss.item()}')
            scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(optimizer); scaler.update()
            total+=loss.item()*len(yb); n+=len(yb); ys.extend(yb.detach().cpu().tolist()); ps.extend(logits.argmax(1).detach().cpu().tolist())
        scheduler.step(); dev_logits,_,_=infer(model,dev,mods,device); devm=metrics_from_logits(dev_logits,ydev)
        train_metrics={'loss':total/max(n,1),'accuracy':float(accuracy_score(ys,ps)),'macro_f1':float(f1_score(ys,ps,labels=list(range(7)),average='macro',zero_division=0))}
        row={'epoch':epoch,'train':train_metrics,'dev':{k:devm[k] for k in ('accuracy','macro_f1','weighted_f1')},'lr':optimizer.param_groups[0]['lr']}; history.append(row)
        scalars={'train/loss':train_metrics['loss'],'train/accuracy':train_metrics['accuracy'],'train/macro_f1':train_metrics['macro_f1'],'dev/accuracy':devm['accuracy'],'dev/macro_f1':devm['macro_f1'],'dev/weighted_f1':devm['weighted_f1'],'train/lr':optimizer.param_groups[0]['lr']}
        if writer:
            for key,value in scalars.items(): writer.add_scalar(key,value,epoch)
        if wandb_run: wandb_run.log(scalars,step=epoch)
        print(f"Epoch {epoch:02d}/{args.epochs} train_loss={train_metrics['loss']:.4f} train_macro={train_metrics['macro_f1']:.4f} dev_acc={devm['accuracy']:.4f} dev_macro={devm['macro_f1']:.4f} dev_weighted={devm['weighted_f1']:.4f}")
        if devm['macro_f1']>best:
            best=devm['macro_f1']; wait=0; torch.save({'model_state_dict':model.state_dict(),'modalities':mods,'fusion_type':args.fusion_type,'loss_type':args.loss_type,'args':vars(args),'best_dev_macro_f1':best,'class_weights':w.cpu()},checkpoint); print(f'  saved best checkpoint (dev Macro-F1={best:.4f})')
        else:
            wait+=1
            if wait>=args.patience: print('Early stopping on Dev Macro-F1.'); break
    # CSV history / reload best
    import pandas as pd
    pd.DataFrame([{'epoch':r['epoch'],'train_loss':r['train']['loss'],'train_accuracy':r['train']['accuracy'],'train_macro_f1':r['train']['macro_f1'],'dev_accuracy':r['dev']['accuracy'],'dev_macro_f1':r['dev']['macro_f1'],'dev_weighted_f1':r['dev']['weighted_f1'],'lr':r['lr']} for r in history]).to_csv(outdir/'training_history.csv',index=False)
    state=torch.load(checkpoint,map_location=device,weights_only=False); model.load_state_dict(state['model_state_dict'])
    dev_logits,_,_=infer(model,dev,mods,device); test_logits,_,gates=infer(model,test,mods,device)
    temperature=fit_temperature(dev_logits,ydev,device)
    raw=save_eval(outdir,'test_uncalibrated',test_logits,ytest,test['text']['clip_ids'],1.0)
    calibrated=save_eval(outdir,'test_calibrated',test_logits,ytest,test['text']['clip_ids'],temperature)
    result={'run_name':run_name,'modalities':mods,'fusion_type':args.fusion_type,'loss_type':args.loss_type,'best_dev_macro_f1':best,'checkpoint':str(checkpoint),'class_counts':dict(zip(EMOTIONS,counts)),'test_uncalibrated':raw,'test_calibrated':calibrated}
    (outdir/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if gates is not None:
        import pandas as pd
        pd.DataFrame(gates.numpy(),columns=[f'gate_{m}' for m in mods]).assign(clip_id=test['text']['clip_ids'] or [str(i) for i in range(len(ytest))]).to_csv(outdir/'modality_gates.csv',index=False)
    print('\nFINAL TEST RESULTS (uncalibrated logits):')
    print(json.dumps(raw,indent=2)); print(f'\nTemperature fitted on DEV only: {temperature:.4f}; calibrated ECE={calibrated["ece"]:.4f}')
    print('Saved reports and plots to:',outdir)
    if writer: writer.close()
    if wandb_run: wandb_run.finish()

if __name__=='__main__': main()
