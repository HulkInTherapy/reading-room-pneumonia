"""P7 full-image, full-training-partition, online FP32 reconstruction helpers."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','2')
import sys,json,random,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np,torch,psutil
from PIL import Image
from torchvision import transforms
from baseline import build_model,MODEL_NAMES,guard,evaluate
SEED=42;EPOCHS=5;BATCH=16
CONFIG=dict(seed=SEED,epochs=EPOCHS,batch_size=BATCH,device='cpu',threads=2,
 training_records=5232,test_records=624,class_mapping={'normal':0,'pneumonia':1},
 frozen_weights='P7 Figures4-8 boundaries',batchnorm='All running statistics update in train mode; affine gradients follow stated frozen/trainable boundaries. Author statistics policy unspecified.',
 augmentation='Fresh per-example horizontal flip0.5, random resized crop scale0.8-1.0 ratio0.75-1.3333, brightness/contrast0.15, grayscale-shared Gaussian pixel noise std0.01. All training examples augmented; no fixed-view caches.',
 normalization={'mean':[.485,.456,.406],'std':[.229,.224,.225]},
 optimizer={'name':'Adam','initial_lr':.001,'betas':[.9,.999],'eps':1e-8,'weight_decay':0},
 lr='Non-AlexNet: halve every3 epochs. AlexNet: first200 optimizer steps0.001 then0.00001.',
 checkpoint='Fixed final epoch5 per model, no selection using any test score',
 precision='FP32 complete image forwards and trainable layers, no activation quantization',
 auxiliary_heads='GoogLeNet and Inception unused auxiliary classifiers disabled',
 test_history='Original benchmark was inspected in earlier experiments; not a blind external holdout',
 known_departures=['Five epochs instead of paper50-200','Author batch size, transform parameters, noise distribution, normalization, BatchNorm-statistics and exact pretrained revision not recovered','AlexNet200-iteration prose versus200-epoch table ambiguity; use literal optimizer steps','Initial224 RGB bilinear cache follows stated resize, but author interpolation/file manifest unverified'])
CONFIG_HASH=hashlib.sha256(json.dumps(CONFIG,sort_keys=True).encode()).hexdigest()
class SharedGrayNoise:
    def __call__(self,x):
        # Keep a grayscale radiograph grayscale before channel normalization.
        return (x+torch.randn(1,x.shape[1],x.shape[2],dtype=x.dtype)*.01).clamp(0,1)
def image_transform(training=False):
    ops=[]
    if training:ops.extend([transforms.RandomHorizontalFlip(.5),transforms.RandomResizedCrop(224,scale=(.8,1),ratio=(.75,4/3)),transforms.ColorJitter(brightness=.15,contrast=.15)])
    ops.append(transforms.ToTensor())
    if training:ops.append(SharedGrayNoise())
    ops.append(transforms.Normalize(CONFIG['normalization']['mean'],CONFIG['normalization']['std']))
    return transforms.Compose(ops)
class Images(torch.utils.data.Dataset):
    def __init__(self,rows,training=False):self.rows=rows;self.transform=image_transform(training)
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        with Image.open(ROOT/r['path']) as im:
            if im.size!=(224,224):raise ValueError(f'Unexpected resized input dimensions: {r["path"]}')
            x=self.transform(im.convert('RGB'))
        return x,int(r['label']),i
def records():
    rs=json.loads((ROOT/'data/manifest.json').read_text())
    train=[r for r in rs if r['split']!='test'];test=[r for r in rs if r['split']=='test']
    assert len(train)==5232 and len(test)==624
    assert [sum(r['label']==c for r in train) for c in [0,1]]==[1349,3883]
    assert [sum(r['label']==c for r in test) for c in [0,1]]==[234,390]
    assert not {r['sha256'] for r in train}&{r['sha256'] for r in test}
    return train,test
def model(name):
    prefix,tail,boundary=build_model(name);prefix.handle.remove();m=prefix.model
    if name=='resnet18':m.fc=tail[3]
    elif name=='densenet121':m.classifier=tail.fc
    elif name=='alexnet':m.classifier=tail
    else:m.fc=tail[4]
    if name=='googlenet':m.aux_logits=False;m.aux1=None;m.aux2=None
    if name=='inception_v3':m.aux_logits=False;m.AuxLogits=None
    allowed={
        'resnet18':('layer4.','fc.'),
        'densenet121':('features.denseblock4.denselayer15.','features.denseblock4.denselayer16.','classifier.'),
        'googlenet':('inception5b.','fc.'),
        'inception_v3':('Mixed_7c.','fc.'),
        'alexnet':('classifier.',)}[name]
    for key,p in m.named_parameters():assert p.requires_grad==key.startswith(allowed),(name,key,p.requires_grad)
    assert not any(module._forward_pre_hooks for module in m.modules()),'Unexpected capture hook remains'
    return m,boundary
def seed_all(seed=SEED):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
def loader(rows,training=False,generator=None):
    return torch.utils.data.DataLoader(Images(rows,training),batch_size=BATCH,shuffle=training,num_workers=0,drop_last=False,pin_memory=False,generator=generator)
def lr_for(name,epoch,step):
    if name=='alexnet':return .001 if step<200 else .00001
    return .001*.5**(epoch//3)
def predict(m,rows):
    m.eval();probs=np.full(len(rows),np.nan,dtype=np.float64);seen=np.zeros(len(rows),dtype=np.int64)
    with torch.inference_mode():
        for x,y,indices in loader(rows):
            guard();z=m(x);assert z.shape==(len(x),2) and torch.isfinite(z).all()
            p=z.softmax(-1)[:,1].numpy();probs[indices.numpy()]=p;seen[indices.numpy()]+=1
    assert (seen==1).all() and np.isfinite(probs).all()
    return probs
def atomic_json(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)
def atomic_torch(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');torch.save(value,temp);temp.replace(path)
