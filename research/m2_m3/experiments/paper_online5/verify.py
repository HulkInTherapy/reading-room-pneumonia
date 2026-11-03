"""Full checkpoint and per-image verification of the fixed-epoch experiment."""
import gc,json,hashlib
import engine as e
import numpy as np,torch
OUT=e.HERE/'results'
def verify():
    train,test=e.records();summary=json.loads((OUT/'summary.json').read_text());assert summary['config_hash']==e.CONFIG_HASH and summary['test_evaluated']
    probabilities=[];checks=[]
    for name in e.MODEL_NAMES:
        done=json.loads((OUT/f'{name}_complete.json').read_text());history=json.loads((OUT/f'{name}_history.json').read_text())
        assert done['config_hash']==e.CONFIG_HASH and done['epochs']==5 and done['train_records']==5232
        assert [h['epoch'] for h in history]==[1,2,3,4,5]
        assert all(h['training_records']==5232 and h['optimizer_steps']==327*h['epoch'] for h in history)
        assert len({h['first_augmented_batch_sha256'] for h in history})==5
        checkpoint=OUT/f'{name}_final.pt';digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest();assert digest==done['checkpoint_sha256']
        m,_=e.model(name)
        initial_counters={n:int(b.num_batches_tracked) for n,b in m.named_modules() if isinstance(b,torch.nn.modules.batchnorm._BatchNorm)}
        m.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
        assert all(int(dict(m.named_modules())[n].num_batches_tracked)==c+1635 for n,c in initial_counters.items())
        p=e.predict(m,test);saved=np.load(OUT/f'{name}_test_probabilities.npy')
        assert p.shape==(624,) and np.allclose(p,saved,atol=1e-7,rtol=1e-6)
        assert np.array_equal(p>=.5,saved>=.5)
        checks.append(dict(model=name,epochs=5,train_records=5232,test_records=624,checkpoint_sha256=digest,max_probability_difference=float(np.max(np.abs(p-saved))),fresh_full_image_forwards=True,all_batchnorm_counters_verified=True))
        probabilities.append(p);del m;gc.collect();print('VERIFIED_MODEL',checks[-1],flush=True)
    p=np.stack(probabilities);y=np.array([r['label'] for r in test]);pred=((p>=.5).sum(0)>=3).astype(int);metrics=e.evaluate(y,p.mean(0),pred)
    for k,value in metrics.items():
        if isinstance(value,list):assert value==summary['metrics'][k]
        else:assert abs(value-summary['metrics'][k])<1e-9,(k,value,summary['metrics'][k])
    tn,fp,fn,tp=np.array(metrics['confusion_matrix']).ravel()
    assert abs(metrics['accuracy']-(tn+tp)/624)<1e-12
    assert abs(metrics['precision']-tp/(tp+fp))<1e-12
    assert abs(metrics['pneumonia_f1']-2*tp/(2*tp+fp+fn))<1e-12
    assert abs(metrics['normal_f1']-2*tn/(2*tn+fp+fn))<1e-12
    receipt=dict(verified=True,models=checks,metrics=metrics,config_hash=e.CONFIG_HASH,test_evaluated=True,selection='Fixed epoch5, no test-driven choices',limitations=e.CONFIG['known_departures'])
    e.atomic_json(OUT/'independent_verification.json',receipt);print('VERIFICATION_PASSED',metrics,flush=True);return receipt
if __name__=='__main__':verify()
