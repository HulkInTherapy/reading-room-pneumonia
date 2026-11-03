"""Real inference for the declared M4 frozen-encoder pilot."""
import gc,json,os
from pathlib import Path
import numpy as np,torch
from PIL import Image
from torchvision import models,transforms
P=Path(__file__).resolve().parents[2]/'m4_m5';OUT=P/'results';DATA=P/'data';WEIGHTS=P/'model_bundle/weights'
os.environ.setdefault('TORCH_HOME',str(P.parent/'torch_cache'))
def ready():
 p=OUT/'summary.json'
 v=OUT/'verification.json'
 return p.exists() and json.loads(p.read_text()).get('completed',False) and v.exists() and json.loads(v.read_text()).get('verified',False) and all((WEIGHTS/f).exists() for f in ['resnet18-f37072fd.pth','deit_tiny_encoder.pt','pspnet_chestxray_best_model_4.pth'])
def segment(im):
 import torchxrayvision as xrv
 model=xrv.baseline_models.chestx_det.PSPNet(cache_dir=str(WEIGHTS)).eval()
 gray=im.convert('L');w,h=gray.size
 a=np.asarray(gray.resize((256,256),Image.Resampling.BILINEAR),dtype=np.float32)
 x=torch.from_numpy(xrv.datasets.normalize(a,255)).unsqueeze(0).unsqueeze(0)
 ids=[model.targets.index(t) for t in ['Left Lung','Right Lung']]
 with torch.inference_mode():mask=(model.model(model.transform((x.repeat(1,3,1,1)+1024)/2048)).sigmoid()[0,ids].max(0).values>.5).numpy()
 del model,x;gc.collect()
 ys,xs=np.where(mask);f=float(mask.mean());fallback=len(xs)==0 or f<.08 or f>.85
 if fallback:box=(0,0,w,h)
 else:
  x0,x1=xs.min()/256*w,(xs.max()+1)/256*w;y0,y1=ys.min()/256*h,(ys.max()+1)/256*h
  dx=(x1-x0)*.1;dy=(y1-y0)*.1;box=(max(0,int(x0-dx)),max(0,int(y0-dy)),min(w,int(x1+dx)),min(h,int(y1+dy)))
 full=im.convert('RGB').resize((224,224),Image.Resampling.BILINEAR)
 crop=im.convert('RGB').crop(box).resize((224,224),Image.Resampling.BILINEAR)
 return full,crop,Image.fromarray(mask.astype('uint8')*255),dict(bbox=list(box),original_size=[w,h],fallback=fallback,mask_fraction=f)
def classify(full,crop):
 import timm
 t=transforms.Compose([transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])]);images=torch.stack([t(full),t(crop)]);out={}
 for arch in ['resnet18','deit_tiny']:
  if arch=='resnet18':
   m=models.resnet18(weights=None);m.load_state_dict(torch.load(WEIGHTS/'resnet18-f37072fd.pth',map_location='cpu',weights_only=True));m.fc=torch.nn.Identity()
  else:
   m=timm.create_model('deit_tiny_patch16_224',pretrained=False,num_classes=0);m.load_state_dict(torch.load(WEIGHTS/'deit_tiny_encoder.pt',map_location='cpu',weights_only=True))
  m.eval()
  with torch.inference_mode():z=torch.cat([m(images[j:j+1]) for j in range(2)],dim=0).numpy()
  del m;gc.collect()
  for j,kind in enumerate(['full','crop']):
   ck=torch.load(OUT/f'{arch}_{kind}.pt',map_location='cpu',weights_only=False)
   head=torch.nn.Linear(z.shape[1],2);head.load_state_dict(ck['state_dict']);head.eval()
   x=torch.tensor((z[j]-np.asarray(ck['mean']))/np.asarray(ck['std']),dtype=torch.float32).unsqueeze(0)
   with torch.inference_mode():out[f'{arch}_{kind}']=float(head(x).softmax(-1)[0,1])
 summary=json.loads((OUT/'summary.json').read_text());conditions={}
 for key,kind in [('A','full'),('B','crop'),('C','full'),('D','crop')]:
  weight=summary['conditions'][key].get('vit_weight',0) if key in ['C','D'] else 0
  prob=(1-weight)*out['resnet18_'+kind]+weight*out['deit_tiny_'+kind]
  conditions[key]=dict(pneumonia_score=prob,prediction='Pneumonia' if prob>=.5 else 'Normal',vit_weight=weight)
 return conditions,out
