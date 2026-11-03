"""Inspect actual final experiment errors without selecting or changing models."""
import csv,json
from pathlib import Path
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
R=Path(__file__).resolve().parent;O=R/'results';rows=list(csv.DictReader((O/'predictions.csv').open()))
record_errors=[r for r in rows if int(float(r['D'])>=.5)!=int(r['label'])]
errors=list({r['sha256']:r for r in record_errors}.values())
chosen=[r for r in errors if r['label']=='0'][:4]+[r for r in errors if r['label']=='1'][:4]
chosen+=( [r for r in errors if r not in chosen][:max(0,8-len(chosen))] )
fig,axs=plt.subplots(2,4,figsize=(12,6.5),facecolor='#f5f4ef')
for a in axs.ravel():a.axis('off')
for a,r in zip(axs.ravel(),chosen):
 p=R/'data/crop'/(r['sha256']+'.png')
 assert p.exists(),str(p)
 a.imshow(Image.open(p),cmap='gray')
 a.set_title(('False positive' if r['label']=='0' else 'False negative')+'\nPneumonia score '+format(float(r['D']),'.2f'),fontsize=10)
fig.tight_layout();fig.savefig(O/'pilot_errors.png',dpi=150);plt.close(fig)
(O/'pilot_error_examples.json').write_text(json.dumps(chosen,indent=2))
print('D errors',len(errors),'examples',len(chosen))
