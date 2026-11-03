"""Fresh bounded 15 epoch run, validation selection, preserved original results."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('VECLIB_MAXIMUM_THREADS','2')
import sys,json,time,gc,copy,random,hashlib,csv,mmap
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from torch import nn
from torchvision import transforms
from PIL import Image
from baseline import build_model,guard,evaluate,MODEL_NAMES,DenseTail
SEED=42;EPOCHS=15;BATCH=16
OUT=HERE/'results';CACHE=HERE/'cache';OUT.mkdir(exist_ok=True);CACHE.mkdir(exist_ok=True)
split=json.loads((HERE/'split.json').read_text());rows=split['records'];ti=np.array(split['train_indices']);vi=np.array(split['validation_indices']);tr=[rows[i] for i in ti];va=[rows[i] for i in vi];yt=np.array([r['label'] for r in tr]);yv=np.array([r['label'] for r in va])
class Noise:
    def __call__(self,x):return (x+torch.randn_like(x)*.01).clamp(0,1)
def transform(view):
    norm=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    return transforms.Compose([transforms.Resize((224,224)),transforms.RandomHorizontalFlip(),transforms.RandomResizedCrop(224,scale=(.8,1)),transforms.ColorJitter(brightness=.15,contrast=.15),transforms.ToTensor(),Noise(),norm]) if view!='clean' else transforms.Compose([transforms.ToTensor(),norm])
def cache(name,prefix,rs,view):
    f=CACHE/f'{name}_{view}.npy';marker=f.with_suffix('.progress.json');meta=f.with_suffix('.meta.json')
    fingerprint=hashlib.sha256(json.dumps([r['sha256'] for r in rs]).encode()).hexdigest()
    if f.exists() and not marker.exists() and meta.exists():
        assert json.loads(meta.read_text())['records_hash']==fingerprint
        return np.load(f,mmap_mode='r')
    begin=json.loads(marker.read_text())['next'] if marker.exists() else 0
    if not marker.exists():marker.write_text(json.dumps({'next':0}))
    tf=transform(view);a=None;viewseed={'clean':0,'aug1':100000,'aug2':200000,'aug3':300000,'validation':0}[view]
    if view=='validation':tf=transform('clean')
    for i in range(begin,len(rs),4):
        guard();images=[]
        for j,r in enumerate(rs[i:i+4],i):
            torch.manual_seed(SEED+j+viewseed)
            with Image.open(ROOT/r['path']) as im:images.append(tf(im.convert('RGB')))
        with torch.inference_mode():z=prefix(torch.stack(images)).numpy().astype(np.float16)
        if a is None:a=np.lib.format.open_memmap(f,mode='r+' if f.exists() else 'w+',dtype='float16',shape=(len(rs),*z.shape[1:]))
        a[i:i+len(z)]=z;marker.write_text(json.dumps({'next':i+len(z)}))
        if i%512==0:print('CACHE',name,view,i,'/',len(rs),'RSS MB',round(__import__('psutil').Process().memory_info().rss/1e6),flush=True)
        time.sleep(.025)
    a.flush();del a;meta.write_text(json.dumps({'records_hash':fingerprint,'dtype':'float16','view':view}));marker.unlink();return np.load(f,mmap_mode='r')
def predict(tail,x):
    tail.eval();probs=[];losses=[]
    with torch.inference_mode():
        for i in range(0,len(x),BATCH):
            xb=torch.from_numpy(np.array(x[i:i+BATCH],dtype=np.float32,copy=True));logits=tail(xb);probs.extend(logits.softmax(-1)[:,1].numpy())
    return np.array(probs)
def val_loss(tail,x,y):
    tail.eval();total=0
    with torch.inference_mode():
        for i in range(0,len(x),BATCH):total+=float(nn.functional.cross_entropy(tail(torch.from_numpy(np.array(x[i:i+BATCH],dtype=np.float32,copy=True))),torch.from_numpy(y[i:i+BATCH]).long(),reduction='sum'))
    return total/len(y)
def bn_diagnostic(name,tail,x):
    h=copy.deepcopy(tail);h.load_state_dict(torch.load(ROOT/'results'/f'{name}_head.pt',weights_only=True,map_location='cpu'));h.eval();ev=predict(h,x)
    order=np.random.default_rng(SEED).permutation(len(yv));batchscores=np.zeros(len(yv))
    for m in h.modules():
        if isinstance(m,nn.modules.batchnorm._BatchNorm):m.train()
    if isinstance(h,DenseTail):h.norm.eval()
    with torch.inference_mode():
        for i in range(0,len(order),32):
            ix=order[i:i+32];batchscores[ix]=h(torch.from_numpy(np.array(x[ix],dtype=np.float32,copy=True))).softmax(-1)[:,1].numpy()
    d=dict(note='Old weights already saw these images; this is a mode-sensitivity diagnostic, not independent validation performance.',stored_running_statistics=evaluate(yv,ev),batch_statistics=evaluate(yv,batchscores),mean_score_shift=float(np.mean(batchscores-ev)),mean_absolute_shift=float(np.mean(np.abs(batchscores-ev))))
    (OUT/f'{name}_bn_diagnostic.json').write_text(json.dumps(d,indent=2));del h;gc.collect();print('BN_DIAGNOSTIC',name,'mean abs score shift',d['mean_absolute_shift'],flush=True)
def fit(name,tail,views,xv):
    torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED);rng=np.random.default_rng(SEED)
    opt=torch.optim.Adam([p for p in tail.parameters() if p.requires_grad],lr=.001);history=[];best=None;start_epoch=0
    resume=OUT/f'{name}_resume.pt'
    if resume.exists():
        # Only locally generated, trusted experiment state is loaded here.
        state=torch.load(resume,map_location='cpu',weights_only=False)
        assert state['seed']==SEED and state['epoch']<=EPOCHS
        tail.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['torch_rng']);rng.bit_generator.state=state['numpy_rng']
        history=state['history'];start_epoch=state['epoch']
        selection=json.loads((OUT/f'{name}_selection.json').read_text())
        best=(selection['metrics']['macro_f1'],selection['metrics']['auc'],-(selection['epoch']-1))
        del state;gc.collect();print('RESUME',name,'after epoch',start_epoch,flush=True)
    for epoch in range(start_epoch,EPOCHS):
        tail.train()
        if isinstance(tail,DenseTail):tail.norm.eval()
        lr=.001*.5**(epoch//3)
        for g in opt.param_groups:g['lr']=lr
        order=rng.permutation(len(yt));loss_sum=0
        current=views[epoch%len(views)]
        for i in range(0,len(order),BATCH):
            guard();ix=order[i:i+BATCH];x=np.array(current[ix],dtype=np.float32,copy=True)
            if name=='alexnet':
                step=epoch*((len(yt)+BATCH-1)//BATCH)+i//BATCH
                for g in opt.param_groups:g['lr']=.001 if step<200 else .00001
            opt.zero_grad(set_to_none=True);loss=nn.functional.cross_entropy(tail(torch.from_numpy(x)),torch.from_numpy(yt[ix]).long());loss.backward();opt.step();loss_sum+=float(loss.detach())*len(ix);time.sleep(.008)
            if i%256==0:
                for v in views:
                    if hasattr(v._mmap,'madvise'):
                        try:v._mmap.madvise(mmap.MADV_DONTNEED)
                        except (OSError,ValueError):pass
            if i%1024==0:print('TRAIN',name,'epoch',epoch+1,'record',i,'/',len(yt),flush=True)
        pv=predict(tail,xv);m=evaluate(yv,pv);vl=val_loss(tail,xv,yv)
        h=dict(epoch=epoch+1,training_view=['clean','aug1','aug2','aug3'][epoch%4],train_loss=loss_sum/len(yt),validation_loss=vl,validation_metrics=m,learning_rate=opt.param_groups[0]['lr']);history.append(h)
        key=(m['macro_f1'],m['auc'],-epoch)
        if best is None or key>best:
            best=key;torch.save(tail.state_dict(),OUT/f'{name}_best.pt');np.save(OUT/f'{name}_validation_probabilities.npy',pv)
            (OUT/f'{name}_selection.json').write_text(json.dumps({'epoch':epoch+1,'criterion':'validation macro F1, AUC, earlier epoch','metrics':m},indent=2))
        state=dict(epoch=epoch+1,model=tail.state_dict(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),numpy_rng=rng.bit_generator.state,history=history,seed=SEED)
        torch.save(state,OUT/f'{name}_resume.pt')
        if epoch+1 in [5,10,15]:torch.save(tail.state_dict(),OUT/f'{name}_epoch{epoch+1}.pt');np.save(OUT/f'{name}_val_epoch{epoch+1}.npy',pv)
        (OUT/f'{name}_history.json').write_text(json.dumps(history,indent=2));print('EPOCH',name,epoch+1,'train_loss',round(h['train_loss'],4),'val_loss',round(vl,4),'val_macro_F1',round(m['macro_f1'],4),'val_auc',round(m['auc'],4),flush=True)
    return tail
def main():
    for name in MODEL_NAMES:
        if (OUT/f'{name}_complete.json').exists():continue
        start=time.time();prefix,tail,boundary=build_model(name)
        views=[cache(name,prefix,tr,v) for v in ['clean','aug1','aug2','aug3']];xv=cache(name,prefix,va,'validation')
        prefix.handle.remove();del prefix;gc.collect();bn_diagnostic(name,tail,xv)
        fit(name,tail,views,xv)
        (OUT/f'{name}_complete.json').write_text(json.dumps(dict(model=name,epochs=EPOCHS,seed=SEED,boundary=boundary,train_n=len(yt),validation_n=len(yv),seconds=time.time()-start),indent=2))
        del tail,views,xv;gc.collect()
        # Our derived training caches only; checkpoints/validation outputs remain.
        for v in ['clean','aug1','aug2','aug3']:
            (CACHE/f'{name}_{v}.npy').unlink()
    report=dict(epoch_validation_ensembles={},selected_validation_ensemble=None,test_evaluated=False)
    for epoch in [5,10,15]:
        p=np.stack([np.load(OUT/f'{n}_val_epoch{epoch}.npy') for n in MODEL_NAMES]);report['epoch_validation_ensembles'][str(epoch)]=evaluate(yv,p.mean(0),((p>=.5).sum(0)>=3).astype(int))
    p=np.stack([np.load(OUT/f'{n}_validation_probabilities.npy') for n in MODEL_NAMES]);report['selected_validation_ensemble']=evaluate(yv,p.mean(0),((p>=.5).sum(0)>=3).astype(int))
    (OUT/'validation_summary.json').write_text(json.dumps(report,indent=2))
    # Only after all model selections are fixed, evaluate the original benchmark.
    pt=[]
    for name in MODEL_NAMES:
        prefix,tail,_=build_model(name);prefix.handle.remove();del prefix;tail.load_state_dict(torch.load(OUT/f'{name}_best.pt',weights_only=True,map_location='cpu'))
        xt=np.load(ROOT/'cache_tail'/f'{name}_test.npy',mmap_mode='r');q=predict(tail,xt);np.save(OUT/f'{name}_test_probabilities.npy',q);pt.append(q);del tail,xt;gc.collect()
    test=split['test_records'];y=np.array([r['label'] for r in test]);pt=np.stack(pt);pred=((pt>=.5).sum(0)>=3).astype(int);score=pt.mean(0)
    result=dict(protocol='Fresh ImageNet initialization; group validation; 15 epochs; batch16; four fixed views; validation-selected checkpoints; fixed0.5 components and majority vote',metrics=evaluate(y,score,pred),validation=report,selected_epochs={n:json.loads((OUT/f'{n}_selection.json').read_text())['epoch'] for n in MODEL_NAMES},old_five_epoch_metrics=json.loads((ROOT/'results/summary.json').read_text())['majority_vote'],warning='Multiple settings/split changed; this comparison does not isolate epoch count. Test was inspected in earlier work; no test-based selection in this run.')
    seen=set();ix=[]
    for i,r in enumerate(test):
        if r['sha256'] not in seen:seen.add(r['sha256']);ix.append(i)
    result['unique_test_metrics']=evaluate(y[ix],score[ix],pred[ix])
    with (OUT/'test_predictions.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['path','sha256','true','predicted','score',*MODEL_NAMES])
        for i,r in enumerate(test):w.writerow([r['path'],r['sha256'],y[i],pred[i],score[i],*pt[:,i]])
    (OUT/'summary.json').write_text(json.dumps(result,indent=2));print('FINAL',json.dumps(result),flush=True)
if __name__=='__main__':main()
