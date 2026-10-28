"""Run the seven MELD modality ablations using the research fusion trainer."""
import argparse, subprocess, sys, json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
EXPERIMENTS=[('text',),('audio',),('video',),('text','audio'),('text','video'),('audio','video'),('text','audio','video')]
def main():
 p=argparse.ArgumentParser(); p.add_argument('--fusion-type',choices=['concat','gated','cross_attention'],default='concat'); p.add_argument('--loss-type',choices=['cross_entropy','weighted_ce','focal'],default='weighted_ce'); p.add_argument('--epochs',type=int,default=30); p.add_argument('--label-smoothing',type=float,default=.1); p.add_argument('--seed',type=int,default=42); a=p.parse_args()
 script=ROOT/'fusion'/'train_research_fusion.py'
 for mods in EXPERIMENTS:
  name='ablation_'+a.fusion_type+'_'+a.loss_type+'_'+'_'.join(mods)
  cmd=[sys.executable,str(script),'--modalities',*mods,'--fusion-type',a.fusion_type,'--loss-type',a.loss_type,'--epochs',str(a.epochs),'--label-smoothing',str(a.label_smoothing),'--seed',str(a.seed),'--run-name',name]
  print('\n'+'='*80+'\nRUNNING: '+name+'\n'+'='*80,flush=True)
  subprocess.run(cmd,cwd=ROOT,check=True)
 summaries=[]
 for mods in EXPERIMENTS:
  name='ablation_'+a.fusion_type+'_'+a.loss_type+'_'+'_'.join(mods)
  path=ROOT/'results'/'research_runs'/name/'summary.json'
  if path.exists():
   obj=json.loads(path.read_text(encoding='utf-8')); r=obj['test_uncalibrated']
   summaries.append({'experiment':name,'modalities':'+'.join(mods),'fusion_type':a.fusion_type,'loss_type':a.loss_type,'accuracy':r['accuracy'],'macro_f1':r['macro_f1'],'weighted_f1':r['weighted_f1'],'ece':r['ece']})
 if summaries:
  out=ROOT/'results'/'research_runs'; pd.DataFrame(summaries).to_csv(out/'ablation_comparison.csv',index=False)
  d=pd.DataFrame(summaries); x=range(len(d)); width=.25; plt.figure(figsize=(13,6)); plt.bar([i-width for i in x],d['accuracy']*100,width,label='Accuracy'); plt.bar(x,d['weighted_f1']*100,width,label='Weighted-F1'); plt.bar([i+width for i in x],d['macro_f1']*100,width,label='Macro-F1'); plt.xticks(list(x),d['modalities'],rotation=25,ha='right'); plt.ylabel('Score (%)'); plt.title('MELD seven-way modality ablation'); plt.legend(); plt.tight_layout(); plt.savefig(out/'ablation_comparison.png',dpi=220); plt.close()
 print('\nAll seven ablations completed. See results/research_runs/ablation_comparison.csv and ablation_comparison.png')
if __name__=='__main__': main()
