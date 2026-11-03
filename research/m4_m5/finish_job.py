"""Durable postchecks and a predeclared training-only mask sanity sheet."""
from pathlib import Path
import time,json,subprocess,sys
P=Path(__file__).resolve().parent;S=json.loads((P/'split.json').read_text());targets=[S['records'][i] for i in S['train_indices'][:6]];made=False
while True:
 try:
  if not made and all((P/'data/masks'/(r['sha256']+'.png')).exists() for r in targets):
   from PIL import Image,ImageDraw
   out=Image.new('RGB',(3*256,6*280),'white');draw=ImageDraw.Draw(out)
   for row,r in enumerate(targets):
    for col,kind in enumerate(['full','masks','crop']):
     with Image.open(P/'data'/kind/(r['sha256']+'.png')) as im:out.paste(im.convert('RGB').resize((256,256)),(col*256,row*280+24))
     draw.text((col*256+5,row*280+5),kind+' | training example '+str(row+1),fill='black')
   out.save(P/'results/training_mask_sanity.png');(P/'results/training_mask_sanity.json').write_text(json.dumps(dict(selection='First six declared training_indices, before observing crop/classification results',records=targets),indent=2));made=True
  q=P/'results/summary.json'
  if q.exists() and json.loads(q.read_text()).get('completed'):
   p=subprocess.run([sys.executable,str(P/'verify_experiment.py')],capture_output=True,text=True);(P/'postcheck.log').write_text(p.stdout+'\n'+p.stderr);raise SystemExit(p.returncode)
 except (OSError,json.JSONDecodeError):pass
 time.sleep(30)
