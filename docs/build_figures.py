"""Rebuild publication figures from measured artifacts; no model training."""
import argparse,json,csv,shutil
from pathlib import Path
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path(__file__).resolve().parents[1]/'research');a=p.parse_args();S=a.source;O=Path(__file__).resolve().parent/'figures';O.mkdir(exist_ok=True)
read=lambda x:json.loads((S/x).read_text())
final=read('m4_m5/results/summary.json');v=read('m4_m5/results/verification.json');assert final['completed'] and v['verified']
old=read('m2_m3/results/summary.json')['majority_vote'];audit=read('m2_m3/experiments/restart15/results/summary.json')['metrics'];online=read('m2_m3/experiments/paper_online5/results/summary.json')['metrics'];new=final['conditions']['D']['metrics']
P=['#667F98','#13A396','#8B69C7','#D46451'];plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','savefig.facecolor':'white'})
def save(fig,name):
 fig.tight_layout();fig.savefig(O/(name+'.pdf'),bbox_inches='tight');fig.savefig(O/(name+'.png'),dpi=220,bbox_inches='tight');plt.close(fig)
labels=['Cached 5','Cached 15','Online 5','Final pilot'];runs=[old,audit,online,new]
f,axes=plt.subplots(1,3,figsize=(12,3.5))
for ax,key,title in zip(axes,['accuracy','specificity','macro_f1'],['Accuracy (%)','Normal recall (%)','Macro F1']):
 values=[m[key]*(100 if key!='macro_f1' else 1) for m in runs];bars=ax.bar(labels,values,color=P)
 for b,z in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,z+.8 if key!='macro_f1' else z+.008,f'{z:.2f}' if key!='macro_f1' else f'{z:.3f}',ha='center',fontsize=13)
 ax.set_ylim(0,105 if key!='macro_f1' else 1.05);ax.set_title(title);ax.tick_params(axis='x',rotation=15);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
save(f,'history')
f,axes=plt.subplots(1,3,figsize=(12,3.6))
for ax,key,title in zip(axes,['accuracy','sensitivity','specificity'],['Accuracy (%)','Pneumonia recall (%)','Normal recall (%)']):
 values=[final['conditions'][k]['metrics'][key]*100 for k in 'ABCD'];bars=ax.bar(list('ABCD'),values,color=P)
 for b,z in zip(bars,values):ax.text(b.get_x()+b.get_width()/2,z+.7,f'{z:.2f}',ha='center',fontsize=13)
 ax.set_ylim(0,106);ax.set_title(title);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
save(f,'ablations')
grid=v['validation_only_selection_audit'];val=[next(r['validation_macro_f1'] for r in grid[k]['grid'] if r['vit_weight']==w) for k,w in [('C',0),('D',0),('C',.25),('D',.5)]]
f,ax=plt.subplots(figsize=(6.2,3.6));x=np.arange(4);ax.bar(x-.18,val,.36,label='Validation',color='#667F98');ax.bar(x+.18,[final['conditions'][k]['metrics']['macro_f1'] for k in 'ABCD'],.36,label='Test',color='#13A396');ax.set(xticks=x,xticklabels=list('ABCD'),ylabel='Macro F1',ylim=(.75,1.01));ax.legend(ncol=2,frameon=False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);save(f,'validation_test')
f,ax=plt.subplots(figsize=(6.4,3.4));x=np.arange(3)
fp=[old['confusion_matrix'][0][1],online['confusion_matrix'][0][1],new['confusion_matrix'][0][1]];fn=[old['confusion_matrix'][1][0],online['confusion_matrix'][1][0],new['confusion_matrix'][1][0]]
ax.bar(x-.18,fp,.36,label='False alarms',color='#D46451');ax.bar(x+.18,fn,.36,label='Missed pneumonia',color='#667F98');ax.set(xticks=x,xticklabels=['Cached 5','Corrected online','Final pilot'],ylabel='Test records');ax.legend(frameon=False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);save(f,'errors_history')
from sklearn.metrics import roc_curve
rs=list(csv.DictReader((S/'m4_m5/results/predictions.csv').open()));y=np.array([int(r['label']) for r in rs]);f,ax=plt.subplots(figsize=(6.2,4.3))
for color,k in zip(P,'ABCD'):
 score=np.array([float(r[k]) for r in rs]);fpr,tpr,_=roc_curve(y,score);ax.plot(fpr,tpr,color=color,lw=2,label=f"{k}: AUC {final['conditions'][k]['metrics']['auc']:.3f}")
ax.plot([0,1],[0,1],'--',color='#bbc3ca',lw=1);ax.set(xlabel='False-positive rate',ylabel='Pneumonia sensitivity',xlim=(0,1),ylim=(0,1.02));ax.legend(loc='lower right',frameon=False);save(f,'roc')
f,axes=plt.subplots(1,4,figsize=(12,3.4))
for ax,k in zip(axes,'ABCD'):
 cm=np.asarray(final['conditions'][k]['metrics']['confusion_matrix']);ax.imshow(cm,cmap='Blues',vmin=0,vmax=390);ax.set_title(k);ax.set_xticks([0,1],['Normal','Pneumonia'],fontsize=12);ax.set_yticks([0,1],['Normal','Pneumonia'],fontsize=12);ax.set_xlabel('Predicted');ax.set_ylabel('True label')
 for i in range(2):
  for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=15,color='white' if cm[i,j]>195 else '#172f43')
save(f,'confusion')
f,axes=plt.subplots(1,2,figsize=(10,3.5))
for color,name in zip(P,['resnet18_full','resnet18_crop','deit_tiny_full','deit_tiny_crop']):
 h=read('m4_m5/results/'+name+'_history.json');xx=[r['epoch'] for r in h];axes[0].plot(xx,[r['loss'] for r in h],'-o',label=name.replace('_',' '),color=color,ms=4);axes[1].plot(xx,[r['validation']['macro_f1'] for r in h],'-o',label=name.replace('_',' '),color=color,ms=4)
for ax in axes:ax.set_xticks([1,2,3,4,5]);ax.set_xlabel('Classifier-head epoch');ax.grid(alpha=.15)
axes[0].set_ylabel('Training cross-entropy');axes[1].set_ylabel('Validation macro F1');axes[1].legend(fontsize=12,frameon=False);save(f,'head_histories')
meta=read('m4_m5/data/crop_metadata.json');fractions=[]
for m in meta.values():
 x0,y0,x1,y1=m['bbox'];w,h=m['original_size'];fractions.append((x1-x0)*(y1-y0)/(w*h))
f,ax=plt.subplots(figsize=(6.2,3.5));ax.hist(np.array(fractions)*100,bins=25,color='#13A396',edgecolor='white');ax.axvline(np.median(fractions)*100,color='#D46451',ls='--',lw=2,label=f'Median {np.median(fractions):.1%}');ax.set(xlabel='Retained original-image area (%)',ylabel='Unique images');ax.legend(frameon=False);save(f,'crop_areas')
for name in ['segmentation_pipeline','training_balanced_mask_sanity','fallback_pipeline','fallback_examples','pilot_errors','baseline_errors','final_demo_desktop']:
 src=S/'m4_m5/results'/(name+'.png')
 if src.exists():shutil.copy2(src,O/(name+'.png'))
# Public sources and measured data are the sole inputs to these figures.
(O.parent/'figure_metadata.json').write_text(json.dumps({'historical':runs,'final':final,'validation_macro_f1':val,'figures_rebuilt_from':'stored experiment CSV/JSON; no training performed'},indent=2))
print('Publication figures built:',O)
