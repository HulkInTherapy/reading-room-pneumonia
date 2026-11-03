from pathlib import Path
import json,time
from PIL import Image,ImageDraw
P=Path(__file__).resolve().parent;S=json.loads((P/'split.json').read_text());train=[S['records'][i] for i in S['train_indices']];targets=[r for label in [0,1] for r in [r for r in train if r['label']==label][:3]]
while not all((P/'data/masks'/(r['sha256']+'.png')).exists() for r in targets):time.sleep(30)
out=Image.new('RGB',(3*256,6*280),'white');draw=ImageDraw.Draw(out)
for row,r in enumerate(targets):
 for col,kind in enumerate(['full','masks','crop']):
  with Image.open(P/'data'/kind/(r['sha256']+'.png')) as im:out.paste(im.convert('RGB').resize((256,256)),(col*256,row*280+24))
  draw.text((col*256+5,row*280+5),kind+' | training '+('normal' if r['label']==0 else 'pneumonia'),fill='black')
out.save(P/'results/training_balanced_mask_sanity.png');(P/'results/training_balanced_mask_sanity.json').write_text(json.dumps(dict(selection='First three declared TRAINING indices per class; no threshold changes',records=targets),indent=2))
