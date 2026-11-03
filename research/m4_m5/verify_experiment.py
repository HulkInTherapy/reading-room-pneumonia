"""Independent arithmetic and artifact checks, no retraining/model selection."""
import json,csv,hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,roc_auc_score,confusion_matrix
P=Path(__file__).resolve().parent;O=P/'results';s=json.loads((O/'summary.json').read_text());assert s['completed'];rows=list(csv.DictReader((O/'predictions.csv').open()));y=np.array([int(r['label']) for r in rows]);checked={}
for key in 'ABCD':
 p=np.array([float(r[key]) for r in rows]);z=p>=.5;m=dict(accuracy=accuracy_score(y,z),sensitivity=recall_score(y,z),specificity=recall_score(1-y,1-z),precision=precision_score(y,z),macro_f1=f1_score(y,z,average='macro'),auc=roc_auc_score(y,p));expected=s['conditions'][key]['metrics'];assert all(abs(v-expected[k])<1e-10 for k,v in m.items());assert confusion_matrix(y,z).tolist()==expected['confusion_matrix'];checked[key]=True
s['protocol'].update(training_records=4164,validation_records=1042,test_records=624,device='CPU',threads=2,precision='FP32',head_optimizer=dict(name='AdamW',learning_rate=.001,weight_decay=.01,batch_size=32,loss='Unweighted cross entropy'),feature_standardization='Training mean/std only; std floor0.001',image_normalization=dict(mean=[.485,.456,.406],std=[.229,.224,.225]),segmentation=dict(input_resolution=256,mask_threshold=.5,margin_per_side=.1,fallback_mask_fraction=[.08,.85],fixed_pretrained=True),primary_metric='Macro F1')
unique={}
for row in rows:unique.setdefault(row['sha256'],row)
urs=list(unique.values());uy=np.array([int(r['label']) for r in urs]);s['unique_image_sensitivity_analysis']={}
for key in 'ABCD':
 up=np.array([float(r[key]) for r in urs]);uz=up>=.5
 s['unique_image_sensitivity_analysis'][key]=dict(records=len(urs),accuracy=accuracy_score(uy,uz),sensitivity=recall_score(uy,uz),specificity=recall_score(1-uy,1-uz),macro_f1=f1_score(uy,uz,average='macro'),auc=roc_auc_score(uy,up))
(O/'summary.json').write_text(json.dumps(s,indent=2))
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in O.glob('*.pt')};assert len(hashes)>=4
meta=json.loads((P/'data/crop_metadata.json').read_text());assert len(meta)==5824
for sha in meta:
 for kind in ['full','crop','masks']:assert (P/'data'/kind/(sha+'.png')).exists()
report=dict(verified=False,scope='Independent CSV metric recomputation and file completeness/hashes; runtime inference separately tested',records=len(rows),conditions=checked,checkpoint_hashes=hashes,crop_unique_images=len(meta),fallbacks=sum(x['fallback'] for x in meta.values()))
split=json.loads((P/'split.json').read_text());audit={}
for name,records in [('train',[split['records'][i] for i in split['train_indices']]),('validation',[split['records'][i] for i in split['validation_indices']]),('test',split['test_records'])]:
 audit[name]={}
 for cls in [0,1]:
  rs=[r for r in records if r['label']==cls];audit[name][str(cls)]=dict(records=len(rs),fallbacks=sum(meta[r['sha256']]['fallback'] for r in rs))
report['crop_audit_by_split_class']=audit
(O/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from PIL import Image
selected=list(meta)[::max(1,len(meta)//6)][:6];fig,axes=plt.subplots(6,3,figsize=(9,15))
for row,sha in enumerate(selected):
 for col,kind in enumerate(['full','masks','crop']):
  axes[row,col].imshow(Image.open(P/'data'/kind/(sha+'.png')),cmap='gray');axes[row,col].axis('off')
  if row==0:axes[row,col].set_title(kind.title())
fig.tight_layout();fig.savefig(O/'segmentation_montage.png',dpi=140);plt.close(fig)
fig,axes=plt.subplots(1,4,figsize=(14,4))
for ax,key in zip(axes,'ABCD'):
 cm=np.array(s['conditions'][key]['metrics']['confusion_matrix']);ax.imshow(cm,cmap='Blues');ax.set_title(key,fontsize=16);ax.tick_params(labelsize=10);ax.set_xticks([0,1],['Normal','Pneumonia']);ax.set_yticks([0,1],['Normal','Pneumonia']);ax.set_xlabel('Predicted',fontsize=12);ax.set_ylabel('Actual',fontsize=12)
 for i in range(2):
  for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=16,color='white' if cm[i,j]>cm.max()/2 else 'black')
fig.tight_layout();fig.savefig(O/'confusion_matrices.png',dpi=140)

first=split['records'][split['train_indices'][0]]['sha256'];fig,axes=plt.subplots(1,3,figsize=(12,4))
for ax,kind,title in zip(axes,['full','masks','crop'],['Full image RGB (224 px)','Predicted lung mask (256 px)','Lung crop RGB (224 px)']):
 ax.imshow(Image.open(P/'data'/kind/(first+'.png')),cmap='gray');ax.set_title(title);ax.axis('off')
fig.tight_layout();fig.savefig(O/'segmentation_pipeline.png',dpi=180);plt.close(fig)
# Reload all four trained heads and verify stored feature-to-probability results.
import os
os.environ['HF_HUB_OFFLINE']='1'
import torch,timm,ast
from torchvision import models
torch.set_num_threads(2)
probs={};validation_probs={};selection_audit={};reload_check={};counts={}
for arch in ['resnet18','deit_tiny']:
 m=models.resnet18(weights=None) if arch=='resnet18' else timm.create_model('deit_tiny_patch16_224',pretrained=True,num_classes=0)
 if arch=='resnet18':m.fc=torch.nn.Identity()
 counts[arch+'_encoder']=sum(p.numel() for p in m.parameters())
 if arch=='deit_tiny':
  bundle=P/'model_bundle/weights';bundle.mkdir(parents=True,exist_ok=True);torch.save(m.state_dict(),bundle/'deit_tiny_encoder.pt')
 del m
 for kind in ['full','crop']:
  ck=torch.load(O/(arch+'_'+kind+'.pt'),map_location='cpu',weights_only=False);z=np.load(P/'data'/(arch+'_'+kind+'.npy'))
  head=torch.nn.Linear(z.shape[1],2);head.load_state_dict(ck['state_dict']);head.eval();counts[arch+'_'+kind+'_head']=sum(p.numel() for p in head.parameters())
  with torch.inference_mode():all_p=head(torch.tensor((z-ck['mean'])/ck['std'],dtype=torch.float32)).softmax(-1)[:,1].numpy()
  probs[arch+'_'+kind]=all_p[len(split['records']):];validation_probs[arch+'_'+kind]=all_p[split['validation_indices']]
  assert np.allclose(ck['mean'],z[split['train_indices']].mean(0),atol=1e-6)
  assert np.allclose(ck['std'],z[split['train_indices']].std(0).clip(.001),atol=1e-6)
  history=json.loads((O/(arch+'_'+kind+'_history.json')).read_text());expected_epoch=max(history,key=lambda r:r['validation']['macro_f1'])['epoch'];assert ck['selected_epoch']==expected_epoch;selection_audit[arch+'_'+kind]=dict(selected_epoch=expected_epoch,training_only_scaler_verified=True)
for key,kind in [('A','full'),('B','crop'),('C','full'),('D','crop')]:
 weight=s['conditions'][key].get('vit_weight',0);p=(1-weight)*probs['resnet18_'+kind]+weight*probs['deit_tiny_'+kind];expected=np.array([float(r[key]) for r in rows]);delta=float(np.max(np.abs(p-expected)));assert delta<2e-5;reload_check[key]=delta
vy=np.array([r['label'] for r in split['records']])[split['validation_indices']]
for key,kind in [('C','full'),('D','crop')]:
 grid=[]
 for weight in [0,.25,.5,.75,1]:
  pv=(1-weight)*validation_probs['resnet18_'+kind]+weight*validation_probs['deit_tiny_'+kind];grid.append(dict(vit_weight=weight,validation_macro_f1=f1_score(vy,pv>=.5,average='macro')))
 chosen=max(grid,key=lambda r:r['validation_macro_f1']);assert chosen['vit_weight']==s['conditions'][key]['vit_weight'];selection_audit[key]=dict(grid=grid,selected_vit_weight=chosen['vit_weight'])
report['validation_only_selection_audit']=selection_audit
report['reloaded_head_max_probability_difference']=reload_check;report['parameter_counts']=counts
(O/'verification.json').write_text(json.dumps(report,indent=2))
mtimes=[p.stat().st_mtime for p in (P/'data/masks').glob('*.png')];phases={};attempts={}
for line in ('\n'.join(q.read_text() for q in [P/'train.log',P/'train256.log'])).splitlines():
 if line.startswith('features '):
  d=ast.literal_eval(line[len('features '):]);key=d['arch']+'_'+d['kind'];phases[key]=d
  attempts.setdefault(key,[])
  if d['complete']==0 or not attempts[key]:attempts[key].append(d)
  else:attempts[key][-1]=d
runtime=dict(mask_file_wall_interval_seconds=max(mtimes)-min(mtimes),mask_file_interval_note='Earliest-to-latest uniform256 mask file mtimes; includes pauses/downloads between masks, excludes initial model/first-shard download',feature_last_logged_elapsed=phases,feature_attempt_last_logged_elapsed=attempts,feature_timing_note='Last logged forward-loop elapsed per attempt; includes retained failed-attempt logs separately; excludes initial model load/download and final unlogged images',head_timing='Not independently captured',parameter_counts=counts)
(P/'runtime_report.json').write_text(json.dumps(runtime,indent=2));print('Reload checks and parameter counts',reload_check,counts)

report['verified']=True
(O/'verification.json').write_text(json.dumps(report,indent=2))
