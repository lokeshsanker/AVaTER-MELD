"""Aggregate completed advanced audio/video backbone test metrics into a comparison table and plot."""
import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def main():
 rows=[]
 for modality in ('audio','video'):
  for p in sorted((ROOT/'results').glob(f'{modality}_advanced_*_test_metrics.json')):
   d=json.loads(p.read_text(encoding='utf-8')); name=p.name.replace(f'{modality}_advanced_','').replace('_test_metrics.json','')
   rows.append({'modality':modality,'backbone':name,**d})
 if not rows:
  print('No advanced backbone test metric files found yet. Train at least one audio/video backbone first.'); return
 out=ROOT/'results'; df=pd.DataFrame(rows); df.to_csv(out/'advanced_backbone_benchmark.csv',index=False); print(df.to_string(index=False))
 for metric in ('accuracy','macro_f1','weighted_f1'):
  plt.figure(figsize=(10,5)); labels=[f'{r.modality}:{r.backbone}' for r in df.itertuples()]; plt.bar(labels,df[metric]*100); plt.ylabel(metric.replace('_',' ').title()+' (%)'); plt.title('Advanced backbone benchmark - '+metric.replace('_',' ').title()); plt.xticks(rotation=25,ha='right'); plt.tight_layout(); plt.savefig(out/f'advanced_backbone_{metric}.png',dpi=220); plt.close()
 print('Saved results/advanced_backbone_benchmark.csv and metric plots.')
if __name__=='__main__': main()
