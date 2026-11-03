"""Execute a submission notebook from actual final run artifacts."""
import json,sys
from pathlib import Path
import nbformat as nb
from nbclient import NotebookClient
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'results'
def main():
    receipt=json.loads((OUT/'independent_verification.json').read_text());assert receipt['verified']
    cells=[]
    def md(s):cells.append(nb.v4.new_markdown_cell(s))
    def code(s):cells.append(nb.v4.new_code_cell(s))
    md('''# Pneumonia baseline reconstruction with online image training

Samarth Shekhar 230042, Meghavi Rao 230044, Rachit Kumar 230128, Harshal Nerpagar 230076.

Base paper: [Chouhan et al. 2020](https://doi.org/10.3390/app10020559). Five ImageNet CNNs, paper trainable boundaries, majority voting. This is our third experiment, with five fixed epochs on all 5,232 training records, fresh augmentation, FP32 full-image forwards and updating BatchNorm running statistics while retaining frozen early weights. Three models trained on the Mac and two in the isolated CPU VPS. The original 624-record test set stays unchanged.

Training ran in the CLI jobs. This notebook executes analysis and independent verification of actual saved artifacts, not simulated training outputs. Five epochs is shorter than the paper50-200 schedules, and author settings remain partly unspecified. No test-driven checkpoint or threshold choice. Crop plus ViT is still an unimplemented M3 proposal.''')
    code('''from pathlib import Path
import sys,json,numpy as np,pandas as pd
ROOT=Path.cwd()
if ROOT.name=='output':ROOT=ROOT.parent
if (ROOT/'m2_m3').is_dir():ROOT=ROOT/'m2_m3'
assert (ROOT/'experiments/paper_online5').is_dir()
EXP=ROOT/'experiments/paper_online5';OUT=EXP/'results'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(EXP))
from engine import MODEL_NAMES,evaluate,CONFIG
summary=json.loads((OUT/'summary.json').read_text())
receipt=json.loads((OUT/'independent_verification.json').read_text())
assert receipt['verified'] and summary['test_evaluated']
print(json.dumps(CONFIG,indent=2))
''')
    md((HERE/'PROTOCOL.txt').read_text())
    md('''## Full executable implementation

The attached engine.py, preflight.py and run.py contain the complete model assembly, image transforms and training loop. Keep RUN_TRAINING=False for saved-results inspection. Reproduction deliberately requires images and official weights, resource checks and a long CPU run. Previously completed checkpoints are retained; use a separate experiment directory for a genuinely new run.''')
    code('''from IPython.display import Code,display
for filename in ['engine.py','preflight.py','run.py']:
    print(filename)
    display(Code((EXP/filename).read_text(),language='python'))
RUN_TRAINING=False
if RUN_TRAINING:
    import subprocess
    subprocess.run([sys.executable,str(EXP/'preflight.py')],check=True)
    subprocess.run([sys.executable,str(EXP/'run.py')],check=True)
''')
    md('''## Independent checkpoint verification

The supplied receipt records the completed CLI reload verification of every full checkpoint on all624 images, including actual epochs/record counts and BatchNorm counters. Below, this notebook independently recalculates metrics from saved per-image scores. Set RECHECK_WEIGHTS=True to deliberately repeat all full-image forwards inside the notebook. A compact distribution omits large checkpoints/images, so full learned-weight rechecking requires reproduction.''')
    code('''RECHECK_WEIGHTS=False  # CLI already completed the full-checkpoint forward verification
weights_present=all((OUT/f'{n}_final.pt').exists() for n in MODEL_NAMES)
if RECHECK_WEIGHTS and weights_present:
    from verify import verify
    verified=verify()
    assert verified['verified']
else:
    print('Completed CLI full-checkpoint verification receipt:',receipt['verified'])
    print('Notebook independently checks saved per-image metrics below. Full forward recheck remains optional.')
manifest=json.loads((ROOT/'data/manifest.json').read_text());test=[r for r in manifest if r['split']=='test']
y=np.array([r['label'] for r in test]);p=np.stack([np.load(OUT/f'{n}_test_probabilities.npy') for n in MODEL_NAMES])
assert p.shape==(5,624) and np.isfinite(p).all()
pred=((p>=.5).sum(0)>=3).astype(int)
recomputed=evaluate(y,p.mean(0),pred)
assert recomputed['confusion_matrix']==summary['metrics']['confusion_matrix']
print('Scores independently recalculated:',recomputed)
''')
    md('''## Paper and measured test results

All local rows use the same624 benchmark records. The test was seen in earlier work and is not a blind external holdout. Several settings changed together, so this comparison does not isolate epoch count. Published numbers are reported claims rather than our independently verified paper implementation.''')
    code('''keys=['accuracy','precision','sensitivity','specificity','pneumonia_f1','normal_f1','macro_f1','auc']
old=json.loads((ROOT/'results/summary.json').read_text())['majority_vote']
audit=json.loads((ROOT/'experiments/restart15/results/summary.json').read_text())['metrics']
paper={'accuracy':.9639,'precision':.9328,'sensitivity':.9962,'auc':.9934}
display(pd.DataFrame([{'run':'Paper reported',**paper},{'run':'Initial cached5',**old},{'run':'Cached audit15',**audit},{'run':'Online corrected5',**summary['metrics']}]).set_index('run')[keys].round(4))
display(pd.DataFrame(summary['model_metrics']).T[keys].round(4))
print('618-unique-image sensitivity analysis:',summary['unique_test_metrics'])
''')
    code('''import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay,roc_curve
fig,axs=plt.subplots(1,2,figsize=(10,4))
ConfusionMatrixDisplay(np.array(summary['metrics']['confusion_matrix']),display_labels=['Normal','Pneumonia']).plot(ax=axs[0],colorbar=False,cmap='Blues')
fpr,tpr,_=roc_curve(y,p.mean(0));axs[1].plot(fpr,tpr,label=f"AUC {summary['metrics']['auc']:.4f}")
axs[1].plot([0,1],[0,1],'--',color='gray');axs[1].set(xlabel='False positive rate',ylabel='Pneumonia recall');axs[1].legend();fig.tight_layout();fig.savefig(OUT/'confusion_and_roc.png',dpi=140);plt.show()
''')
    md('''## Actual training histories

Each model completed exactly five epochs and1635 optimizer steps. Each epoch processes all5232 records once. These are training losses on freshly augmented images, not validation or test accuracy curves. Final epoch-five weights were fixed before test evaluation.''')
    code('''fig,ax=plt.subplots(figsize=(9,4))
for n in MODEL_NAMES:
    h=json.loads((OUT/f'{n}_history.json').read_text())
    assert len(h)==5 and all(a['training_records']==5232 for a in h)
    ax.plot([a['epoch'] for a in h],[a['training_loss'] for a in h],label=n)
ax.set(xlabel='Epoch',ylabel='Training cross entropy');ax.legend();fig.tight_layout();fig.savefig(OUT/'training_curves.png',dpi=140);plt.show()
''')
    md((ROOT/'HYPOTHESIS_BEFORE_IMPLEMENTATION.md').read_text())
    md('''## Limits

Matching paper partition/architecture does not establish exact author equivalence. Fresh augmentation severity, normalization, BatchNorm policy and classifier details remain declared choices. Five epochs is intentionally truncated. Single-center pediatric images, uncertain patient identifiers, repeated test records and one seed limit generalization. We do not select models or thresholds from test scores, claim a new state of the art or use predictions as clinical diagnoses.''')
    notebook=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'CV Python','language':'python','name':'python3'},'language_info':{'name':'python'}})
    NotebookClient(notebook,timeout=900,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    path=ROOT/'output/M2_Final_Online_Baseline.ipynb';nb.write(notebook,path)
    print('EXECUTED_NOTEBOOK',path,flush=True)
if __name__=='__main__':main()
