"""Execute artifact-audit cells and save their real captured output in a notebook."""
import io,contextlib,traceback
from pathlib import Path
import nbformat
R=Path(__file__).resolve().parent
nb=nbformat.v4.new_notebook();nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
nb.cells=[nbformat.v4.new_markdown_cell('''# Final milestone: lung crop and CNN–ViT pilot

The VPS ran pretrained feature extraction and five epochs of classifier-head optimization. This notebook executes artifact inspection and independent score recomputation locally. It does not claim to rerun training in this notebook. Full training code and logs are included in the project. M2 and this frozen-feature pilot use different protocols.
''')]
codes=[
'''from pathlib import Path
import json, csv, hashlib
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
ROOT=Path.cwd()
if not (ROOT/'results/summary.json').exists(): ROOT=ROOT.parent
summary=json.loads((ROOT/'results/summary.json').read_text())
assert summary['completed']
assert json.loads((ROOT/'results/verification.json').read_text())['verified']
print(json.dumps(summary['protocol'],indent=2))''',
'''rows=list(csv.DictReader((ROOT/'results/predictions.csv').open()))
y=np.array([int(r['label']) for r in rows])
for key,condition in summary['conditions'].items():
 p=np.array([float(r[key]) for r in rows]);z=(p>=.5).astype(int)
 computed={'accuracy':accuracy_score(y,z),'sensitivity':recall_score(y,z),'specificity':recall_score(1-y,1-z),'precision':precision_score(y,z,zero_division=0),'macro_f1':f1_score(y,z,average='macro'),'auc':roc_auc_score(y,p)}
 for metric,value in computed.items():assert abs(value-condition['metrics'][metric])<1e-10
 print(key,json.dumps(computed),'confusion',confusion_matrix(y,z).tolist())
print('All declared metrics independently recomputed from',len(rows),'test predictions.')''',
'''for path in sorted((ROOT/'results').glob('*_history.json')):
 history=json.loads(path.read_text())
 assert len(history)==5
 print(path.name,'epochs',len(history),'validation macro F1',[round(e['validation']['macro_f1'],4) for e in history])
print('Crop audit:',summary.get('crop_audit'))
print('Paired crop effect interval:',summary.get('paired_macro_f1_B_minus_A_95CI'))''',
'''for p in sorted((ROOT/'results').glob('*.pt')):
 print(p.name,hashlib.sha256(p.read_bytes()).hexdigest())
print('The training source is run_experiment.py; use the recorded VPS environment and split for reproduction.')''']
ns={};original=Path.cwd()
import os
os.chdir(R)
for i,code in enumerate(codes,1):
 cell=nbformat.v4.new_code_cell(code,execution_count=i);out=io.StringIO()
 with contextlib.redirect_stdout(out):exec(compile(code,'audit_cell','exec'),ns)
 cell.outputs=[nbformat.v4.new_output('stream',name='stdout',text=out.getvalue())];nb.cells.append(cell)
os.chdir(original)
import base64
for filename,title in [('training_histories.png','Training and validation histories'),('roc.png','ROC curves'),('confusion_matrices.png','Confusion matrices'),('segmentation_pipeline.png','Actual training-image preprocessing'),('pilot_errors.png','Actual crop-hybrid errors')]:
 path=R/'results'/filename
 if path.exists():
  cell=nbformat.v4.new_markdown_cell('## '+title+'\n\n!['+title+'](attachment:'+filename+')')
  cell.attachments={filename:{'image/png':base64.b64encode(path.read_bytes()).decode()}};nb.cells.append(cell)
nb.cells.append(nbformat.v4.new_markdown_cell('## Training implementation (documented source)\n\n```python\n'+(R/'run_experiment.py').read_text()+'\n```'))
nbformat.write(nb,R/'output/Final_M4_M5_Results.ipynb')
