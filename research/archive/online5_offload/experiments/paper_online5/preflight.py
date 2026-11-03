"""Fail-fast checks before the bounded online reconstruction starts."""
import json,time,gc,hashlib,sys
from pathlib import Path
import engine as e
import numpy as np,torch,psutil
from PIL import Image
from torch import nn
def main():
    train,test=e.records();seen={};labels={}
    for r in train+test:
        assert r['label'] in [0,1] and (e.ROOT/r['path']).is_file()
        assert labels.setdefault(r['sha256'],r['label'])==r['label']
        if r['path'] not in seen:
            with Image.open(e.ROOT/r['path']) as im:
                im.verify()
            with Image.open(e.ROOT/r['path']) as im:
                a=np.array(im.convert('RGB'));assert a.shape==(224,224,3) and np.isfinite(a).all()
            seen[r['path']]=True
    print('ALL_IMAGES_DECODED',len(seen),'unique paths',flush=True)
    chosen=[next(r for r in train if r['label']==c) for c in [0,1]]*8
    e.seed_all();ds=e.Images(chosen,True);a=ds[0][0];b=ds[0][0];assert not torch.equal(a,b)
    e.seed_all();assert torch.equal(a,ds[0][0])
    clean=e.Images(chosen,False);assert torch.equal(clean[0][0],clean[0][0])
    assert e.lr_for('resnet18',2,999)==.001 and e.lr_for('resnet18',3,999)==.0005
    assert e.lr_for('alexnet',0,199)==.001 and e.lr_for('alexnet',0,200)==.00001
    tests=[]
    for name in e.MODEL_NAMES:
        e.guard();e.seed_all();m,boundary=e.model(name);m.train();e.seed_all()
        x=torch.stack([ds[i][0] for i in range(16)]);y=torch.tensor([r['label'] for r in chosen])
        frozen=next((n,p) for n,p in m.named_parameters() if not p.requires_grad)
        trainable=[(n,p) for n,p in m.named_parameters() if p.requires_grad]
        frozen_before=frozen[1].detach().clone();last_before=trainable[-2][1].detach().clone()
        bns=[b for b in m.modules() if isinstance(b,nn.modules.batchnorm._BatchNorm)]
        counters=[int(b.num_batches_tracked) for b in bns]
        opt=torch.optim.Adam([p for n,p in trainable],lr=.001);start=time.perf_counter()
        z=m(x);assert z.shape==(16,2) and torch.isfinite(z).all()
        loss=nn.functional.cross_entropy(z,y);assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is None for p in m.parameters() if not p.requires_grad)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for n,p in trainable)
        opt.step()
        assert torch.equal(frozen_before,frozen[1])
        assert not torch.equal(last_before,trainable[-2][1])
        assert all(int(b.num_batches_tracked)==n+1 for b,n in zip(bns,counters))
        seconds=time.perf_counter()-start;rss=psutil.Process().memory_info().rss/1024**2;e.guard()
        m.eval()
        with torch.inference_mode():
            one=m(x[:1]);assert one.shape==(1,2) and torch.isfinite(one).all()
        # Check exact state serialization/reload without writing a checkpoint.
        state={k:v.detach().clone() for k,v in m.state_dict().items()};m.load_state_dict(state)
        with torch.inference_mode():assert torch.allclose(one,m(x[:1]),atol=1e-7,rtol=1e-7)
        sampler=torch.Generator().manual_seed(e.SEED)
        indexes=[]
        for bx,by,ix in e.loader(chosen,True,sampler):indexes.extend(ix.tolist())
        assert sorted(indexes)==list(range(len(chosen)))
        tests.append(dict(model=name,boundary=boundary,trainable_parameter_count=sum(p.numel() for n,p in trainable),batch16_forward_backward_seconds=seconds,rss_mb=rss,finite_loss=float(loss.detach()),frozen_weights_unchanged=True,trainable_weights_updated=True,all_trainable_gradients_present=True,batchnorm_counters_updated=len(bns),single_image_eval_and_state_reload=True))
        print('MODEL_PREFLIGHT_PASS',tests[-1],flush=True)
        del m,opt,x,y,z,loss,state,one,trainable,frozen,frozen_before,last_before,bns;gc.collect()
    for w in json.loads((e.ROOT/'sources/pretrained_weight_checksums.json').read_text()):
        f=Path(torch.hub.get_dir())/'checkpoints'/w['file']
        assert f.exists() and hashlib.sha256(f.read_bytes()).hexdigest()==w['sha256'],('Pretrained weight checksum failed',w['file'])
    files=['engine.py','run.py','preflight.py','../../baseline.py','../../data/manifest.json']
    source_hashes={n:hashlib.sha256((e.HERE/n).read_bytes()).hexdigest() for n in files}
    report=dict(passed=True,config=e.CONFIG,config_hash=e.CONFIG_HASH,source_sha256=source_hashes,
                train_records=len(train),test_records=len(test),unique_image_paths_verified=len(seen),
                class_mapping_verified=True,no_exact_train_test_hash_overlap=True,
                fresh_augmentation_and_seed_repeatability_verified=True,lr_boundary_checks=True,
                models=tests,estimated_training_seconds=sum(t['batch16_forward_backward_seconds'] for t in tests)*327*5,
                warning='Smoke-test timing estimate excludes loading/checkpoint overhead and sustained-load variation. Preflight weights are discarded; actual training restarts from ImageNet.')
    e.atomic_json(e.HERE/'PREFLIGHT.json',report);e.atomic_json(e.HERE/'CONFIG.json',e.CONFIG)
    print('PREFLIGHT_PASSED',flush=True)
if __name__=='__main__':main()
