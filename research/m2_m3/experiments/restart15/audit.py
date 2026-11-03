import sys,json,re,gc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import StratifiedGroupKFold
from baseline import build_model
HERE=Path(__file__).resolve().parent
def group(r):
    name=r.get('source_path') or ''
    p=re.search(r'person(\d+)',name)
    if p:return 'pneumonia-person-'+p.group(1)
    p=re.search(r'(NORMAL2-IM-\d+|IM-\d+)',name)
    return 'normal-'+p.group(1) if p else 'hash-'+r['sha256']
def make_split():
    records=json.loads((ROOT/'data/manifest.json').read_text());unique={}
    for r in records:
        if r['split']!='test':
            if r['sha256'] in unique:assert unique[r['sha256']]['label']==r['label']
            unique.setdefault(r['sha256'],r)
    rows=list(unique.values());y=np.array([r['label'] for r in rows]);groups=np.array([group(r) for r in rows])
    a,b=next(StratifiedGroupKFold(5,shuffle=True,random_state=42).split(np.zeros(len(y)),y,groups))
    assert not set(groups[a])&set(groups[b])
    test=[r for r in records if r['split']=='test'];assert not {r['sha256'] for r in rows}&{r['sha256'] for r in test}
    split=dict(records=rows,train_indices=a.tolist(),validation_indices=b.tolist(),test_records=test,seed=42,group_definition='conservative person number / normal filename prefix / hash fallback')
    (HERE/'split.json').write_text(json.dumps(split,indent=2))
    counts={n:dict(total=len(i),normal=int((y[i]==0).sum()),pneumonia=int((y[i]==1).sum()),groups=len(set(groups[i]))) for n,i in [('train',a),('validation',b)]}
    (HERE/'split_audit.json').write_text(json.dumps(dict(counts=counts,original_non_test=5232,unique_non_test=len(rows),hash_overlap=0,group_overlap=0),indent=2));print(counts,flush=True)
    return split
def numerical_check(split):
    rows=split['records'];chosen=[r for c in [0,1] for r in rows if r['label']==c][:0]
    chosen=[]
    for c in [0,1]:chosen.extend([r for r in rows if r['label']==c][:8])
    t=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    result=[]
    for name in ['resnet18','densenet121','googlenet','inception_v3','alexnet']:
        p,h,_=build_model(name);h.load_state_dict(torch.load(ROOT/'results'/f'{name}_head.pt',map_location='cpu',weights_only=True));h.eval();xs=[]
        for r in chosen:
            with Image.open(ROOT/r['path']) as im:xs.append(t(im.convert('RGB')))
        with torch.inference_mode():z=p(torch.stack(xs))
        p.handle.remove()
        with torch.inference_mode():a=h(z).softmax(-1)[:,1].numpy();b=h(z.half().float()).softmax(-1)[:,1].numpy()
        result.append(dict(model=name,n=len(a),max_probability_difference=float(np.abs(a-b).max()),mean_probability_difference=float(np.abs(a-b).mean()),decision_changes=int(((a>=.5)!=(b>=.5)).sum())))
        del p,h,z,xs;gc.collect();print(result[-1],flush=True)
    (HERE/'quantization_audit.json').write_text(json.dumps(result,indent=2))
    return result
if __name__=='__main__':numerical_check(make_split())
