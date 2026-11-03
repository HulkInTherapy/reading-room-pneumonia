"""Five epochs on all5232 paper-training images with online augmentation."""
import time,json,gc,hashlib,csv,random,platform,argparse
from pathlib import Path
import engine as e
import numpy as np,torch,psutil
from torch import nn
OUT=e.HERE/'results';OUT.mkdir(exist_ok=True)
def status(phase,model=None,epoch=None,batch=None,**extra):
    e.atomic_json(e.HERE/'STATUS.json',dict(phase=phase,current_model=model,current_epoch=epoch,current_batch=batch,updated=time.strftime('%Y-%m-%d %H:%M:%S'),epochs_per_model=e.EPOCHS,completed_models=[n for n in e.MODEL_NAMES if (OUT/f'{n}_complete.json').exists()],**extra))
def train_model(name,rows):
    if (OUT/f'{name}_complete.json').exists():
        done=json.loads((OUT/f'{name}_complete.json').read_text());assert done['config_hash']==e.CONFIG_HASH
        assert (OUT/f'{name}_final.pt').exists();return
    e.seed_all();m,boundary=e.model(name);e.seed_all()
    generator=torch.Generator().manual_seed(e.SEED)
    opt=torch.optim.Adam([p for p in m.parameters() if p.requires_grad],lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0)
    history=[];steps=0;begin=0;resume=OUT/f'{name}_resume.pt';start=time.time()
    if resume.exists():
        state=torch.load(resume,map_location='cpu',weights_only=False)
        assert state['config_hash']==e.CONFIG_HASH
        m.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);begin=state['completed_epochs'];steps=state['steps'];history=state['history']
        torch.set_rng_state(state['torch_rng']);generator.set_state(state['sampler_rng']);np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
        del state;gc.collect();print('RESUME',name,'after',begin,'epochs',flush=True)
    for epoch in range(begin,e.EPOCHS):
        m.train();loader=e.loader(rows,True,generator);loss_sum=0.;seen=np.zeros(len(rows),dtype=np.int16);epoch_start=time.time();correct=0;first_hash=None
        for batch,(x,y,indexes) in enumerate(loader):
            e.guard();assert torch.equal(y,torch.tensor([rows[i]['label'] for i in indexes.tolist()]))
            assert torch.isfinite(x).all() and x.dtype==torch.float32
            lr=e.lr_for(name,epoch,steps)
            for g in opt.param_groups:g['lr']=lr
            opt.zero_grad(set_to_none=True);logits=m(x);assert logits.shape==(len(x),2) and torch.isfinite(logits).all()
            loss=nn.functional.cross_entropy(logits,y);assert torch.isfinite(loss),'Nonfinite training loss'
            loss.backward()
            assert all(p.grad is None for p in m.parameters() if not p.requires_grad)
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
            opt.step();steps+=1;loss_sum+=float(loss.detach())*len(y);correct+=int((logits.argmax(1)==y).sum());seen[indexes.numpy()]+=1
            if first_hash is None:first_hash=hashlib.sha256(x.numpy().tobytes()).hexdigest()
            if batch%32==0:
                status('training',name,epoch+1,batch+1,steps=steps,elapsed_seconds=time.time()-start,rss_mb=round(psutil.Process().memory_info().rss/1024**2))
                print('TRAIN',name,'epoch',epoch+1,'batch',batch+1,'/',len(loader),'loss',round(float(loss.detach()),5),flush=True)
            time.sleep(.008)
        assert (seen==1).all(), 'Every training record must occur exactly once per epoch'
        h=dict(epoch=epoch+1,training_records=int(seen.sum()),optimizer_steps=steps,training_loss=loss_sum/len(rows),training_accuracy_augmented=correct/len(rows),last_learning_rate=opt.param_groups[0]['lr'],first_augmented_batch_sha256=first_hash,seconds=time.time()-epoch_start)
        history.append(h)
        assert len({a['first_augmented_batch_sha256'] for a in history})==len(history)
        state=dict(config_hash=e.CONFIG_HASH,model=m.state_dict(),optimizer=opt.state_dict(),completed_epochs=epoch+1,steps=steps,history=history,torch_rng=torch.get_rng_state(),sampler_rng=generator.get_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
        e.atomic_torch(resume,state);e.atomic_json(OUT/f'{name}_history.json',history);del state
        print('EPOCH_COMPLETED',name,h,flush=True)
    assert len(history)==5 and steps==1635
    e.atomic_torch(OUT/f'{name}_final.pt',m.state_dict())
    e.atomic_json(OUT/f'{name}_complete.json',dict(model=name,epochs=5,train_records=len(rows),boundary=boundary,config_hash=e.CONFIG_HASH,checkpoint_sha256=hashlib.sha256((OUT/f'{name}_final.pt').read_bytes()).hexdigest(),seconds=time.time()-start,history=history))
    del m,opt;gc.collect();e.guard()
def main(names=e.MODEL_NAMES):
    pre=json.loads((e.HERE/'PREFLIGHT.json').read_text());assert pre['passed'] and pre['config_hash']==e.CONFIG_HASH
    for filename,digest in pre['source_sha256'].items():assert hashlib.sha256((e.HERE/filename).read_bytes()).hexdigest()==digest,('Source changed after preflight',filename)
    train,test=e.records()
    for name in names:train_model(name,train)
    if not all((OUT/f'{name}_complete.json').exists() for name in e.MODEL_NAMES):
        status('assigned_models_complete_other_models_pending');return
    # Fixed checkpoints exist before any final test forward in this experiment.
    status('final_test_evaluation');allp=[]
    for name in e.MODEL_NAMES:
        m,_=e.model(name);m.load_state_dict(torch.load(OUT/f'{name}_final.pt',map_location='cpu',weights_only=True));p=e.predict(m,test)
        np.save(OUT/f'{name}_test_probabilities.npy',p);allp.append(p);del m;gc.collect()
    p=np.stack(allp);y=np.array([r['label'] for r in test]);prediction=((p>=.5).sum(0)>=3).astype(int)
    result=dict(protocol=e.CONFIG,config_hash=e.CONFIG_HASH,metrics=e.evaluate(y,p.mean(0),prediction),model_metrics={n:e.evaluate(y,p[i]) for i,n in enumerate(e.MODEL_NAMES)},test_evaluated=True,checkpoint_epoch=5,checkpoint_selection='Fixed final epoch5, all models; no test selection',environment={'platform':platform.platform(),'torch':torch.__version__},warning='Five epochs remains a shortened reconstruction. Several author settings are unspecified. Benchmark test was seen previously.')
    unique=[];hashes=set()
    for i,r in enumerate(test):
        if r['sha256'] not in hashes:hashes.add(r['sha256']);unique.append(i)
    result['unique_test_metrics']=e.evaluate(y[unique],p.mean(0)[unique],prediction[unique])
    with (OUT/'test_predictions.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['path','sha256','true_label','predicted_label','mean_score',*e.MODEL_NAMES])
        for i,r in enumerate(test):w.writerow([r['path'],r['sha256'],y[i],prediction[i],p[:,i].mean(),*p[:,i]])
    e.atomic_json(OUT/'summary.json',result);status('training_and_test_complete_verification_pending');print('FINAL_METRICS',result['metrics'],flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--models',nargs='+',choices=e.MODEL_NAMES,default=e.MODEL_NAMES);args=parser.parse_args()
    try:main(args.models)
    except Exception as ex:
        status('failed',error=repr(ex));raise
