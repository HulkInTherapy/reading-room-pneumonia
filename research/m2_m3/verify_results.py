"""Independently recompute every headline from per-image saved probabilities."""
import json,hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,roc_auc_score,confusion_matrix,balanced_accuracy_score
ROOT=Path(__file__).resolve().parent
def verify(require_heads=True):
    manifest=json.loads((ROOT/'data/manifest.json').read_text())
    test=[r for r in manifest if r['split']=='test'];y=np.array([r['label'] for r in test])
    summary=json.loads((ROOT/'results/summary.json').read_text());names=summary['models']
    p=np.stack([np.load(ROOT/'results'/f'{n}_probabilities.npy') for n in names])
    assert p.shape==(5,624) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    pr=(p>=.5).sum(0)>=3;s=p.mean(0)
    tn,fp,fn,tp=confusion_matrix(y,pr).ravel()
    independently=dict(accuracy=float(accuracy_score(y,pr)),sensitivity=float(recall_score(y,pr)),specificity=float(tn/(tn+fp)),precision=float(precision_score(y,pr)),pneumonia_f1=float(f1_score(y,pr)),macro_f1=float(f1_score(y,pr,average='macro')),normal_f1=float(f1_score(y,pr,pos_label=0)),auc=float(roc_auc_score(y,s)),balanced_accuracy=float(balanced_accuracy_score(y,pr)))
    for k,v in independently.items():assert abs(v-summary['majority_vote'][k])<1e-12,(k,v)
    # Predictions from the learned saved heads must match the logged output.
    import torch
    heads_present=all((ROOT/'cache_tail'/f'{n}_test.npy').exists() and (ROOT/'results'/f'{n}_head.pt').exists() for n in names)
    if require_heads and not heads_present:raise RuntimeError('Saved heads or test features missing: run reproduction first.')
    for n in (names if heads_present else []):
        x=np.load(ROOT/'cache_tail'/f'{n}_test.npy',mmap_mode='r')
        state=torch.load(ROOT/'results'/f'{n}_head.pt',map_location='cpu',weights_only=True)
        constructor=globals().get('build_model')
        if constructor is None:
            from baseline import build_model as constructor
        prefix,h,_=constructor(n)
        prefix.handle.remove()
        del prefix
        h.load_state_dict(state);h.eval();res=[]
        with torch.inference_mode():
            for i in range(0,len(x),8):res.extend(h(torch.from_numpy(np.array(x[i:i+8],dtype=np.float32,copy=True))).softmax(-1)[:,1].numpy())
        assert np.allclose(res,np.load(ROOT/'results'/f'{n}_probabilities.npy'),atol=1e-7)
    unique=[];seen=set()
    for i,r in enumerate(test):
        if r['sha256'] not in seen:unique.append(i);seen.add(r['sha256'])
    u=np.array(unique)
    unique_metrics=dict(n=len(u),accuracy=float(accuracy_score(y[u],pr[u])),macro_f1=float(f1_score(y[u],pr[u],average='macro')),auc=float(roc_auc_score(y[u],s[u])))
    files=[ROOT/'baseline.py',ROOT/'data/manifest.json',ROOT/'results/test_predictions.csv',ROOT/'HYPOTHESIS_BEFORE_IMPLEMENTATION.md']
    receipt=dict(status='passed',independently_recomputed=independently,confusion_matrix=[[int(tn),int(fp)],[int(fn),int(tp)]],saved_heads_predictions_verified=heads_present,unique_test_sensitivity_analysis=unique_metrics,sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    (ROOT/'results/verification.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2));return receipt
if __name__=='__main__':verify()
