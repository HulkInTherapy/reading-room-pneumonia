"""PRELIMINARY DIAGNOSTIC ONLY: fully frozen-backbone approximation of P7.

Measured protocol departures are recorded in results, never hidden. CPU feature
extraction, two threads, batch four, pauses; no GPU or parallel model training.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','2')
import argparse, copy, csv, gc, hashlib, json, time
from pathlib import Path
import numpy as np
import psutil
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score

ROOT=Path(__file__).resolve().parent
torch.set_num_threads(2);torch.set_num_interop_threads(1)
SEED=42
MODEL_NAMES=['resnet18','densenet121','googlenet','inception_v3','alexnet']
def guard():
    p=psutil.Process();vm=psutil.virtual_memory()
    if p.memory_info().rss>2.6*1024**3 or vm.available<650*1024**2:
        raise RuntimeError('Resource guard: process >2.6 GiB or system available <650 MiB. Resume cache later.')
    if psutil.disk_usage(str(ROOT)).free<700*1024**2:raise RuntimeError('Disk guard <700 MiB free')

def build_model(name):
    weights=getattr(models,name)(weights='DEFAULT')
    if name=='alexnet':
        feature=nn.Sequential(weights.features,weights.avgpool,nn.Flatten())
        head=weights.classifier;head[-1]=nn.Linear(4096,2)
    elif name=='densenet121':
        feature=nn.Sequential(weights.features,nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)),nn.Flatten())
        head=nn.Linear(weights.classifier.in_features,2)
    else:
        d=weights.fc.in_features;weights.fc=nn.Identity()
        if hasattr(weights,'aux_logits'): weights.aux_logits=False
        feature=weights;head=nn.Linear(d,2)
    feature.eval()
    for p in feature.parameters():p.requires_grad=False
    return feature,head

def load_records():
    records=json.loads((ROOT/'data/manifest.json').read_text())
    train=[r for r in records if r['split']!='test']
    test=[r for r in records if r['split']=='test']
    if {r['sha256'] for r in train}&{r['sha256'] for r in test}:raise RuntimeError('Exact duplicate leakage across train/test')
    assert len(train)==5232 and len(test)==624
    return train,test

def features(name,feature,records,view):
    cache=ROOT/'cache';cache.mkdir(exist_ok=True)
    file=cache/f'{name}_{view}.npy';progress=cache/f'{name}_{view}.progress.json'
    if file.exists() and not progress.exists():return np.load(file,mmap_mode='r')
    normal=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    clean=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),normal])
    aug=transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),transforms.RandomResizedCrop(224,scale=(.8,1.0)),transforms.ColorJitter(brightness=.15,contrast=.15),transforms.ToTensor(),normal])
    transform=aug if view=='aug' else clean
    state=json.loads(progress.read_text()) if progress.exists() else {'next':0}
    output=None
    batch_size=4
    for start in range(state['next'],len(records),batch_size):
        guard();xs=[]
        for idx,r in enumerate(records[start:start+batch_size],start):
            torch.manual_seed(SEED+idx+(100000 if view=='aug' else 0))
            with Image.open(ROOT/r['path']) as im:xs.append(transform(im.convert('RGB')))
        with torch.inference_mode(): z=feature(torch.stack(xs)).cpu().numpy().astype(np.float32)
        if output is None:
            if file.exists():output=np.lib.format.open_memmap(file,mode='r+')
            else:output=np.lib.format.open_memmap(file,mode='w+',dtype='float32',shape=(len(records),z.shape[1]))
        output[start:start+len(z)]=z
        progress.write_text(json.dumps({'next':start+len(z)}))
        if start%256==0:print(name,view,start,'/',len(records),'RSS MB',round(psutil.Process().memory_info().rss/1e6),flush=True)
        time.sleep(.035)
    output.flush();del output;progress.unlink()
    return np.load(file,mmap_mode='r')

def evaluate(y,prob,pred=None):
    pred=(prob>=.5).astype(int) if pred is None else pred
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(n=len(y),accuracy=float(accuracy_score(y,pred)),sensitivity=float(recall_score(y,pred)),specificity=float(tn/(tn+fp)),precision=float(precision_score(y,pred,zero_division=0)),pneumonia_f1=float(f1_score(y,pred)),macro_f1=float(f1_score(y,pred,average='macro')),normal_f1=float(f1_score(y,pred,pos_label=0)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),auc=float(roc_auc_score(y,prob)),confusion_matrix=[[int(tn),int(fp)],[int(fn),int(tp)]])

def fit(name,head,clean,aug,y):
    torch.manual_seed(SEED);np.random.seed(SEED)
    opt=torch.optim.Adam(head.parameters(),lr=.001)
    history=[]
    # Fixed budget, no test-based early stopping. Paper has unspecified ResNet epochs.
    epochs=50 if name=='googlenet' else 100
    total_steps=400 if name=='alexnet' else None
    rng=np.random.default_rng(SEED);steps=0
    for epoch in range(epochs):
        head.train();losses=[]
        order=rng.permutation(len(y))
        if name=='alexnet':
            # 200 iterations at 1e-3 then 200 at 1e-5; latter count assumption.
            order=order[:32*4]
        lr=(.001 if steps<200 else .00001) if name=='alexnet' else .001*(.5**(epoch//3))
        for g in opt.param_groups:g['lr']=lr
        for start in range(0,len(order),32):
            guard();idx=order[start:start+32]
            use_aug=rng.random(len(idx))<.5
            x=np.array(clean[idx],copy=True);x[use_aug]=aug[idx[use_aug]]
            xb=torch.from_numpy(x);yb=torch.from_numpy(y[idx]).long()
            opt.zero_grad(set_to_none=True);loss=nn.functional.cross_entropy(head(xb),yb);loss.backward();opt.step()
            losses.append(float(loss.detach()));steps+=1
            time.sleep(.005 if name!='alexnet' else .025)
        history.append(dict(epoch=epoch+1,steps=steps,loss=float(np.mean(losses)),learning_rate=lr))
        if epoch%10==0:print('Train',name,history[-1],flush=True)
        if total_steps and steps>=total_steps:break
    head.eval();return head,history

def predict(head,x):
    out=[]
    with torch.inference_mode():
        for i in range(0,len(x),32):out.extend(head(torch.from_numpy(np.array(x[i:i+32],copy=True))).softmax(-1)[:,1].numpy().tolist())
    return np.array(out)

def main(names=MODEL_NAMES):
    started=time.time();train,test=load_records();y=np.array([r['label'] for r in train]);yt=np.array([r['label'] for r in test])
    out=ROOT/'results';out.mkdir(exist_ok=True)
    for name in names:
        if all(f.exists() for f in [out/f'{name}.json',out/f'{name}_head.pt',out/f'{name}_probabilities.npy',ROOT/'cache'/f'{name}_test.npy']):continue
        model_started=time.time()
        torch.manual_seed(SEED);feature,head=build_model(name)
        weight_sha=hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in feature.parameters())).hexdigest()
        clean=features(name,feature,train,'train');aug=features(name,feature,train,'aug');xt=features(name,feature,test,'test')
        del feature;gc.collect()
        head,history=fit(name,head,clean,aug,y)
        prob=predict(head,xt);metric=evaluate(yt,prob)
        torch.save(head.state_dict(),out/f'{name}_head.pt')
        np.save(out/f'{name}_probabilities.npy',prob)
        (out/f'{name}.json').write_text(json.dumps(dict(model=name,metrics=metric,history=history,pretrained_backbone_sha256=weight_sha,seed=SEED,seconds=time.time()-model_started),indent=2))
        print('RESULT',name,json.dumps(metric),flush=True)
        del head,clean,aug,xt;gc.collect()
    if all((out/f'{n}.json').exists() for n in MODEL_NAMES):
        probs=np.stack([np.load(out/f'{n}_probabilities.npy') for n in MODEL_NAMES])
        votes=(probs>=.5).sum(0);pred=(votes>=3).astype(int);score=probs.mean(0)
        result=dict(majority_vote=evaluate(yt,score,pred),score_definition='AUC uses mean probability. Class labels use 3-of-5 majority vote.',majority_class=evaluate(yt,np.ones(len(yt))),train_n=len(train),test_n=len(test),models=MODEL_NAMES,seed=SEED,device='cpu',threads=2,feature_batch=4,protocol='P7 independent reconstruction; two cached augmentation views; classifier-only training; unspecified details declared in checklist',total_seconds=time.time()-started)
        rng=np.random.default_rng(SEED);boot={k:[] for k in ['accuracy','sensitivity','specificity','macro_f1','auc']}
        for _ in range(1000):
            ix=rng.integers(0,len(yt),len(yt));m=evaluate(yt[ix],score[ix],pred[ix])
            for k in boot:boot[k].append(m[k])
        result['bootstrap_95_ci']={k:np.quantile(v,[.025,.975]).tolist() for k,v in boot.items()}
        (out/'summary.json').write_text(json.dumps(result,indent=2))
        with (out/'test_predictions.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['path','sha256','true_label','predicted_label','mean_pneumonia_probability','pneumonia_votes',*MODEL_NAMES])
            for i,r in enumerate(test):writer.writerow([r['path'],r['sha256'],yt[i],pred[i],score[i],votes[i],*probs[:,i]])
        print('ENSEMBLE',json.dumps(result),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--models',nargs='+',default=MODEL_NAMES);a=parser.parse_args();main(a.models)
