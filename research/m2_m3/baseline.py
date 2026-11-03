"""P7 Figures 4-8 reconstruction; five-epoch budget is explicitly declared.
Cached frozen prefixes, trainable stated late blocks, CPU two threads only.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('VECLIB_MAXIMUM_THREADS','2')
import gc,json,time,csv,argparse
from pathlib import Path
import numpy as np,psutil,torch
from torch import nn
from torchvision import models,transforms
from PIL import Image
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score,roc_auc_score,confusion_matrix,balanced_accuracy_score
ROOT=Path(__file__).resolve().parent
torch.set_num_threads(2)
if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
MODEL_NAMES=['resnet18','densenet121','googlenet','inception_v3','alexnet'];SEED=42;EPOCHS=5
def guard():
    if psutil.Process().memory_info().rss>2.6*1024**3 or psutil.virtual_memory().available<650*1024**2:raise RuntimeError('Memory resource guard')
    if psutil.disk_usage(str(ROOT)).free<700*1024**2:raise RuntimeError('Disk resource guard')
class Captured(Exception):pass
class Prefix(nn.Module):
    def __init__(self,model,target):
        super().__init__();self.model=model;self.value=None;self.handle=target.register_forward_pre_hook(self.capture)
    def capture(self,module,inputs):
        x=inputs[0];self.value=torch.cat(x,1) if isinstance(x,list) else x;raise Captured()
    def forward(self,x):
        try:self.model(x)
        except Captured:
            z=self.value;self.value=None;return z
        raise RuntimeError('Prefix target not reached')
class DenseTail(nn.Module):
    def __init__(self,m):
        super().__init__();b=m.features.denseblock4;self.layer15=b.denselayer15;self.layer16=b.denselayer16;self.norm=m.features.norm5;self.fc=nn.Linear(1024,2)
    def forward(self,x):
        x=torch.cat([x,self.layer15(x)],1);x=torch.cat([x,self.layer16(x)],1)
        return self.fc(nn.functional.adaptive_avg_pool2d(torch.relu(self.norm(x)),(1,1)).flatten(1))
def build_model(name):
    torch.manual_seed(SEED);m=getattr(models,name)(weights='DEFAULT')
    if hasattr(m,'aux_logits'):m.aux_logits=False
    for p in m.parameters():p.requires_grad=False
    if name=='resnet18':target=m.layer4;tail=nn.Sequential(target,nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Linear(512,2));boundary='layer4 + fc'
    elif name=='densenet121':target=m.features.denseblock4.denselayer15;tail=DenseTail(m);boundary='denseblock4.denselayer15 + denselayer16 + classifier'
    elif name=='googlenet':target=m.inception5b;tail=nn.Sequential(target,nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Dropout(.2),nn.Linear(1024,2));boundary='inception5b + fc'
    elif name=='inception_v3':target=m.Mixed_7c;tail=nn.Sequential(target,nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Dropout(.5),nn.Linear(2048,2));boundary='Mixed_7c + fc'
    else:
        m.classifier=nn.Identity();target=m.classifier;tail=nn.Sequential(nn.Linear(9216,512),nn.ReLU(),nn.Dropout(.5),nn.Linear(512,2));boundary='9216->512->2 replacement classifier'
    prefix=Prefix(m,target).eval()
    for p in tail.parameters():p.requires_grad=True
    if isinstance(tail,DenseTail):
        for p in tail.norm.parameters():p.requires_grad=False
    return prefix,tail,boundary
def load_records():
    rows=json.loads((ROOT/'data/manifest.json').read_text());tr=[r for r in rows if r['split']!='test'];te=[r for r in rows if r['split']=='test']
    assert len(tr)==5232 and len(te)==624 and not ({r['sha256'] for r in tr}&{r['sha256'] for r in te});return tr,te
def features(name,prefix,records,view):
    cache=ROOT/'cache_tail';cache.mkdir(exist_ok=True);f=cache/f'{name}_{view}.npy';marker=f.with_suffix('.progress.json')
    if f.exists() and not marker.exists():return np.load(f,mmap_mode='r')
    norm=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    t=transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),transforms.RandomResizedCrop(224,scale=(.8,1)),transforms.ColorJitter(brightness=.15,contrast=.15),transforms.ToTensor(),norm]) if view=='aug' else transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),norm])
    start=json.loads(marker.read_text())['next'] if marker.exists() else 0;out=None
    if not marker.exists():marker.write_text(json.dumps({'next':0}))
    for i in range(start,len(records),4):
        guard();xs=[]
        for j,r in enumerate(records[i:i+4],i):
            torch.manual_seed(SEED+j+(100000 if view=='aug' else 0))
            with Image.open(ROOT/r['path']) as im:xs.append(t(im.convert('RGB')))
        with torch.inference_mode():z=prefix(torch.stack(xs)).numpy().astype(np.float16)
        if out is None:out=np.lib.format.open_memmap(f,mode='r+' if f.exists() else 'w+',dtype='float16',shape=(len(records),*z.shape[1:]))
        out[i:i+len(z)]=z;marker.write_text(json.dumps({'next':i+len(z)}))
        if i%512==0:print(name,view,i,'/',len(records),'RSS MB',round(psutil.Process().memory_info().rss/1e6),flush=True)
        time.sleep(.025)
    out.flush();del out;marker.unlink();return np.load(f,mmap_mode='r')
def evaluate(y,prob,pred=None):
    pred=(prob>=.5).astype(int) if pred is None else pred;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(n=len(y),accuracy=float(accuracy_score(y,pred)),sensitivity=float(recall_score(y,pred)),specificity=float(tn/(tn+fp)),precision=float(precision_score(y,pred,zero_division=0)),pneumonia_f1=float(f1_score(y,pred)),normal_f1=float(f1_score(y,pred,pos_label=0)),macro_f1=float(f1_score(y,pred,average='macro')),balanced_accuracy=float(balanced_accuracy_score(y,pred)),auc=float(roc_auc_score(y,prob)),confusion_matrix=[[int(tn),int(fp)],[int(fn),int(tp)]])
def fit(name,tail,clean,aug,y):
    torch.manual_seed(SEED);rng=np.random.default_rng(SEED);opt=torch.optim.Adam([p for p in tail.parameters() if p.requires_grad],lr=.001);hist=[];steps=0
    for epoch in range(EPOCHS):
        tail.train()
        if isinstance(tail,DenseTail):tail.norm.eval()
        lr=.001*.5**(epoch//3)
        for g in opt.param_groups:g['lr']=lr
        losses=[];order=rng.permutation(len(y))
        for i in range(0,len(y),8):
            guard();ix=order[i:i+8];a=rng.random(len(ix))<.5;x=np.array(clean[ix],dtype=np.float32,copy=True);x[a]=aug[ix[a]].astype(np.float32)
            opt.zero_grad(set_to_none=True);loss=nn.functional.cross_entropy(tail(torch.from_numpy(x)),torch.from_numpy(y[ix]).long());loss.backward();opt.step();losses.append(float(loss.detach()));steps+=1;time.sleep(.008)
            if i%1024==0:print('Training',name,'epoch',epoch+1,'record',i,'/',len(y),flush=True)
        hist.append(dict(epoch=epoch+1,steps=steps,loss=float(np.mean(losses)),learning_rate=lr));print('EPOCH',name,hist[-1],flush=True)
    tail.eval();return tail,hist
def predict(tail,x):
    p=[]
    with torch.inference_mode():
        for i in range(0,len(x),8):p.extend(tail(torch.from_numpy(np.array(x[i:i+8],dtype=np.float32,copy=True))).softmax(-1)[:,1].numpy())
    return np.array(p)
def main(names=MODEL_NAMES):
    train,test=load_records();y=np.array([r['label'] for r in train]);yt=np.array([r['label'] for r in test]);out=ROOT/'results';out.mkdir(exist_ok=True)
    for name in names:
        if all(f.exists() for f in [out/f'{name}.json',out/f'{name}_head.pt',out/f'{name}_probabilities.npy',ROOT/'cache_tail'/f'{name}_test.npy']):continue
        started=time.time();prefix,tail,boundary=build_model(name);clean=features(name,prefix,train,'train');aug=features(name,prefix,train,'aug');xt=features(name,prefix,test,'test')
        prefix.handle.remove();del prefix;gc.collect();tail,h=fit(name,tail,clean,aug,y);prob=predict(tail,xt);met=evaluate(yt,prob)
        torch.save(tail.state_dict(),out/f'{name}_head.pt');np.save(out/f'{name}_probabilities.npy',prob)
        (out/f'{name}.json').write_text(json.dumps(dict(model=name,metrics=met,history=h,seed=SEED,trainable_boundary=boundary,epochs=EPOCHS,seconds=time.time()-started,protocol='Figures 4-8 late-block fine-tuning; five-epoch bounded reconstruction'),indent=2));print('RESULT',name,json.dumps(met),flush=True)
        del clean,aug,xt,tail;gc.collect()
        for view in ['train','aug']:(ROOT/'cache_tail'/f'{name}_{view}.npy').unlink()
    if all((out/f'{n}.json').exists() for n in MODEL_NAMES):
        pp=np.stack([np.load(out/f'{n}_probabilities.npy') for n in MODEL_NAMES]);votes=(pp>=.5).sum(0);pred=(votes>=3).astype(int);score=pp.mean(0)
        result=dict(majority_vote=evaluate(yt,score,pred),majority_class=evaluate(yt,np.ones(len(yt))),models=MODEL_NAMES,train_n=len(train),test_n=len(test),seed=SEED,device='cpu',threads=2,feature_batch=4,training_batch=8,epochs=EPOCHS,protocol='Late-block fine-tuning; five epochs; fixed cached augmentation; float16 frozen activations',score_definition='Labels: majority vote. AUC: mean probability.')
        rng=np.random.default_rng(SEED);boot={k:[] for k in ['accuracy','sensitivity','specificity','macro_f1','auc']}
        for _ in range(1000):
            ix=rng.integers(0,len(yt),len(yt));e=evaluate(yt[ix],score[ix],pred[ix])
            for k in boot:boot[k].append(e[k])
        result['bootstrap_95_ci']={k:np.quantile(v,[.025,.975]).tolist() for k,v in boot.items()};(out/'summary.json').write_text(json.dumps(result,indent=2))
        with (out/'test_predictions.csv').open('w') as f:
            w=csv.writer(f);w.writerow(['path','sha256','true_label','predicted_label','mean_pneumonia_probability','pneumonia_votes',*MODEL_NAMES])
            for i,r in enumerate(test):w.writerow([r['path'],r['sha256'],yt[i],pred[i],score[i],votes[i],*pp[:,i]])
        print('ENSEMBLE',json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',default=MODEL_NAMES);a=p.parse_args();main(a.models)
