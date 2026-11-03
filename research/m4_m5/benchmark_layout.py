import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import torch,time,json,gc
from pathlib import Path
import numpy as np
from PIL import Image
import torchxrayvision as xrv
torch.set_num_threads(2);P=Path(__file__).resolve().parent;root=P.parent
r=json.loads((root/'m2_m3/experiments/restart15/split.json').read_text())['records'][0]
a=np.asarray(Image.open(root/'m2_m3'/r['path']).convert('L').resize((256,256)),dtype=np.float32)
x=torch.from_numpy(xrv.datasets.normalize(a,255))[None,None];m=xrv.baseline_models.chestx_det.PSPNet().eval();x=m.transform((x.repeat(1,3,1,1)+1024)/2048)
out={};logits={}
for layout in ['nchw','channels_last']:
 if layout=='channels_last':m.model=m.model.to(memory_format=torch.channels_last);x=x.contiguous(memory_format=torch.channels_last)
 times=[]
 with torch.inference_mode():
  for i in range(4):
   t=time.time();y=m.model(x);elapsed=time.time()-t
   if i:times.append(elapsed)
  logits[layout]=y.clone()
 out[layout]=times
ids=[m.targets.index(t) for t in ['Left Lung','Right Lung']];ma=logits['nchw'].sigmoid()[0,ids].max(0).values>.5;mb=logits['channels_last'].sigmoid()[0,ids].max(0).values>.5
out['max_logit_delta']=float((logits['nchw']-logits['channels_last']).abs().max());out['mask_disagreement_pixels']=int((ma!=mb).sum());out['mask_iou']=float((ma&mb).sum()/(ma|mb).sum());(P/'layout_benchmark.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
