"""Select first unique errors per class after all model choices are frozen."""
from pathlib import Path
import csv,json
P=Path(__file__).resolve().parent;rows=list(csv.DictReader((P/'results/predictions.csv').open()));chosen=[];used=set()
for label in [0,1]:
 count=0
 for i,r in enumerate(rows):
  if int(r['label'])==label and int(float(r['D'])>=.5)!=label and r['sha256'] not in used:
   chosen.append(dict(test_index=i,error_type='false_positive' if label==0 else 'false_negative',**r));used.add(r['sha256']);count+=1
   if count==4:break
for i,r in enumerate(rows):
 if len(chosen)>=8:break
 if int(float(r['D'])>=.5)!=int(r['label']) and r['sha256'] not in used:chosen.append(dict(test_index=i,error_type='false_positive' if int(r['label'])==0 else 'false_negative',**r));used.add(r['sha256'])
(P/'results/error_examples.json').write_text(json.dumps(dict(selection='First four unique D-error images per true class, then remaining first unique errors to at most eight; no retraining/model selection',cases=chosen),indent=2))
(P/'error_image_files.txt').write_text('\n'.join(f'{kind}/{r["sha256"]}.png' for r in chosen for kind in ['full','crop','masks'])+'\n')
print('Selected error cases:',len(chosen))
