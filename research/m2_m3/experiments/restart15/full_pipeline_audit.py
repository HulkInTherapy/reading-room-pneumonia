"""Read-only model diagnosis: fresh forwards on every test image, no tuning."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys,json,gc,time,re,collections
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
import numpy as np,torch
from torchvision import transforms
from PIL import Image
from baseline import build_model,evaluate,MODEL_NAMES,guard
from audit import group
OUT=HERE/'results';summary=json.loads((OUT/'summary.json').read_text());split=json.loads((HERE/'split.json').read_text())
rows=split['test_records'];manifest=json.loads((ROOT/'data/manifest.json').read_text());original=[r for r in manifest if r['split']=='test']
assert [(r['path'],r['sha256'],r['label']) for r in rows]==[(r['path'],r['sha256'],r['label']) for r in original]
assert len(rows)==624 and collections.Counter(r['label'] for r in rows)=={0:234,1:390}
y=np.array([r['label'] for r in rows]);tf=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
allp=[];checks=[]
for name in MODEL_NAMES:
    start=time.time();prefix,tail,_=build_model(name)
    tail.load_state_dict(torch.load(OUT/f'{name}_best.pt',map_location='cpu',weights_only=True));tail.eval()
    values=[];xt=np.load(ROOT/'cache_tail'/f'{name}_test.npy',mmap_mode='r');feature_delta=0.
    for i in range(0,len(rows),4):
        guard();images=[]
        for r in rows[i:i+4]:
            with Image.open(ROOT/r['path']) as im:
                assert im.size==(224,224);images.append(tf(im.convert('RGB')))
        with torch.inference_mode():z=prefix(torch.stack(images)).half().float()
        feature_delta=max(feature_delta,float(np.max(np.abs(z.numpy()-np.array(xt[i:i+len(images)],dtype=np.float32)))))
        # Prefix target is shared with tail; disable capture while evaluating it.
        prefix.handle.remove()
        with torch.inference_mode():values.extend(tail(z).softmax(-1)[:,1].numpy())
        # Reinstall the same capture hook for the next fresh image batch.
        if name=='resnet18':target=prefix.model.layer4
        elif name=='densenet121':target=prefix.model.features.denseblock4.denselayer15
        elif name=='googlenet':target=prefix.model.inception5b
        elif name=='inception_v3':target=prefix.model.Mixed_7c
        else:target=prefix.model.classifier
        prefix.handle=target.register_forward_pre_hook(prefix.capture)
        if i%128==0:print('FRESH_TEST_AUDIT',name,i,'/',len(rows),flush=True)
        time.sleep(.025)
    prefix.handle.remove();p=np.array(values);saved=np.load(OUT/f'{name}_test_probabilities.npy');difference=float(np.max(np.abs(p-saved)))
    assert np.allclose(p,saved,atol=3e-5,rtol=1e-4),(name,difference)
    assert np.array_equal(p>=.5,saved>=.5)
    checks.append(dict(model=name,images=len(p),max_probability_difference=difference,max_cached_feature_difference=feature_delta,decision_changes=0,seconds=time.time()-start,test_metrics=evaluate(y,p)))
    allp.append(p);del prefix,tail,xt,z;gc.collect();print('MODEL_AUDITED',name,checks[-1],flush=True)
p=np.stack(allp);pred=((p>=.5).sum(0)>=3).astype(int);metrics=evaluate(y,p.mean(0),pred)
assert metrics['confusion_matrix']==summary['metrics']['confusion_matrix']
parts={'train':[split['records'][i] for i in split['train_indices']],'validation':[split['records'][i] for i in split['validation_indices']],'test':rows}
overlap={a+'/'+b:len({group(r) for r in parts[a]}&{group(r) for r in parts[b]}) for a,b in [('train','validation'),('train','test'),('validation','test')]}
view_diagnostics={}
for name in MODEL_NAMES:
    h=json.loads((OUT/f'{name}_history.json').read_text());selection=json.loads((OUT/f'{name}_selection.json').read_text())
    view_diagnostics[name]=dict(selected_epoch=selection['epoch'],selected_training_view=h[selection['epoch']-1]['training_view'],mean_validation_macro_f1_by_view={v:float(np.mean([e['validation_metrics']['macro_f1'] for e in h if e['training_view']==v])) for v in ['clean','aug1','aug2','aug3']})
report=dict(verified=True,test_records=624,test_order_labels_hashes_match_manifest=True,models=checks,ensemble=metrics,view_diagnostics=view_diagnostics,filename_group_proxy_intersections=overlap,proxy_warning='Shared pneumonia person numbers mainly occur across bacterial/viral filenames and may be reused counters. They do not prove shared patients. No exact source hash overlaps non-test/test. Patient-independent test evaluation has NOT been certified.',threshold_selected_on_test=False,weights_modified=False,note='Every original test image was forwarded freshly through its normalized image input and ImageNet prefix; no cached test feature was used to produce these predictions. Existing caches were compared only for diagnosis.')
(OUT/'full_pipeline_audit.json').write_text(json.dumps(report,indent=2));print('FULL_PIPELINE_AUDIT_PASSED',flush=True)
