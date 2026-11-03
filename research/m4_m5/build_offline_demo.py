"""Recorded evidence gallery, explicitly distinguished from live inference."""
import json,html
from pathlib import Path
R=Path(__file__).resolve().parent;O=R/'output';O.mkdir(exist_ok=True)
blocks=[]
for p in sorted((R/'results').glob('demo_*recorded.json')):
 d=json.loads(p.read_text());b='<article><h2>'+html.escape(p.stem.replace('_',' '))+'</h2>'
 if 'conditions' in d:
  b+=''.join('<figure><img src="'+d[k]+'"><figcaption>'+label+'</figcaption></figure>' for k,label in [('full_image','Full input'),('mask_image','Predicted mask'),('crop_image','Crop')]);b+='<pre>'+html.escape(json.dumps(d['conditions'],indent=2))+'</pre>'
 else:
  b+='<p>'+html.escape(d['prediction'])+' · pneumonia score '+str(round(100*d['pneumonia_score'],1))+'%</p>'
  b+=''.join('<figure><img src="'+d[k]+'"><figcaption>'+label+'</figcaption></figure>' for k,label in [('input_image','Input'),('heatmap','ResNet18 Grad-CAM')])
 blocks.append(b+'</article>')
(O/'Offline_Demo_Evidence.html').write_text('<!doctype html><meta charset="utf-8"><title>Recorded model evidence</title><style>body{background:#f5f4ef;color:#132736;font:18px Arial;margin:40px}article{padding:24px;background:white;margin:25px 0}figure{display:inline-block;margin:12px}img{width:224px;height:224px;object-fit:contain;background:#111}pre{font-size:14px;white-space:pre-wrap}</style><h1>Recorded model evidence</h1><p>Offline backup for the presentation. These are saved responses from actual model runs. This page does not perform new inference.</p>'+''.join(blocks))
