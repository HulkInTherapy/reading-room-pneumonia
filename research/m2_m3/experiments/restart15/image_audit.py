import json
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
rows=json.loads((ROOT/'data/manifest.json').read_text());stats={}
for r in rows:
    s='non_test' if r['split']!='test' else 'test';key=f'{s}_class{r["label"]}'
    with Image.open(ROOT/r['path']) as im:a=np.asarray(im.convert('L'),dtype=np.float32)/255
    rim=np.concatenate([a[:16].ravel(),a[-16:].ravel(),a[16:-16,:16].ravel(),a[16:-16,-16:].ravel()])
    stats.setdefault(key,[]).append([r['width'],r['height'],r['width']/r['height'],float(a.mean()),float(a.std()),float(rim.mean()),float(a[40:-40,40:-40].mean())])
names=['original_width','original_height','aspect_ratio','mean_intensity','intensity_std','border_mean','center_mean'];out={}
for key,values in stats.items():
    a=np.array(values);out[key]={n:dict(mean=float(a[:,i].mean()),median=float(np.median(a[:,i])),q25=float(np.quantile(a[:,i],.25)),q75=float(np.quantile(a[:,i],.75))) for i,n in enumerate(names)}
(HERE/'image_distribution_audit.json').write_text(json.dumps(out,indent=2))
for k,v in out.items():print(k,'mean brightness',round(v['mean_intensity']['mean'],3),'std',round(v['intensity_std']['mean'],3),'border',round(v['border_mean']['mean'],3),'original median size',round(v['original_width']['median']),round(v['original_height']['median']))
