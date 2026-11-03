"""Create an executed notebook from real artifacts and executable baseline code."""
from pathlib import Path
import json
import nbformat as nb
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parent
cells=[]
def md(s):cells.append(nb.v4.new_markdown_cell(s))
def code(s):cells.append(nb.v4.new_code_cell(s))
md('''# Reading the Reading Room\n## M2 five-CNN baseline reconstruction and M3 hypothesis

Samarth Shekhar 230042, Meghavi Rao 230044, Rachit Kumar 230128, Harshal Nerpagar 230076.

Base paper: [Chouhan et al. 2020](https://doi.org/10.3390/app10020559). This is an independent reconstruction of the late-block boundaries in Figures4-8 using torchvision, with a bounded FIVE-epoch schedule. It differs from the stated50/100-epoch runs and AlexNet iteration schedule. The corrected run actually used an8 GiB Apple M2, one CPU model at a time. Earlier fully frozen probes are archived separately and are not the final ensemble. This executed notebook audits measured outputs and learned tails. Set MODE to `reproduce` to download data and train missing artifacts; the prepared Mac resumes existing complete models.

The proposed lung-focused crop is not implemented or measured here. M3 requires submitting a prediction first. No clinical, global novelty or state-of-the-art claim is made.''')
code('''from pathlib import Path
import os, sys, json, platform
ROOT = Path.cwd()
if ROOT.name == 'output': ROOT = ROOT.parent
if not (ROOT/'baseline.py').exists() and (ROOT/'m2_m3').exists(): ROOT = ROOT/'m2_m3'
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
MODE = 'inspect_saved'  # change to 'reproduce' only for a deliberate full run
print('Execution mode:', MODE)
print('Platform:', platform.platform())
print('Source code and real saved results directory:', ROOT)
''')
md('''## Assignment alignment

Stage 2 selects one base paper and checks reproducibility. Stage 3 reproduces its baseline. M2 submits a notebook with reproduced results. M3 submits one change, a numerical prediction and mechanism before implementing that change, plus an introduction in the report and presentation. M4 tests the change and ablations. We include sensitivity/recall, AUC and limitations as explicitly required.''')
md((ROOT/'REPRODUCIBILITY_CHECKLIST.md').read_text())
md('''## Dataset preparation code

Source parquet SHA256 checksums are verified. Only one shard is downloaded at a time. We cache the paper's initial RGB 224x224 bilinear resize as lossless PNG to keep disk usage bounded. Source hashes and original filenames stay in the manifest. Redundant task-owned parquet caches are removed after extraction.''')
prep=(ROOT/'prepare_data.py').read_text().split("if __name__=='__main__':")[0].replace('Path(__file__).resolve().parent','Path.cwd()')
code(prep)
code("manifest_file = ROOT/'data/manifest.json'\ndata_ready = manifest_file.exists() and all((ROOT/r['path']).exists() for r in json.loads(manifest_file.read_text()))\nif MODE == 'reproduce' and not data_ready:\n    prepare()\nelse:\n    print('Saved-results inspection or existing verified image cache retained.')")
md('''## Actual dataset audit

The paper training partition merges original train and val: 5,232 training records. Test has 624 records. Checkpoint and decision thresholds are not selected on the test set. Exact hashes do not certify patient independence.''')
audit_source=(ROOT/'audit.py').read_text().split("if __name__=='__main__':")[0].replace('Path(__file__).resolve().parent','Path.cwd()')
code(audit_source)
code("a = audit()\nprint(json.dumps(a, indent=2))")
md('''## Full baseline training code

Frozen early prefixes, trainable late blocks/classifiers from Figures4-8, Adam/cross entropy and majority voting. No lung-crop code is present. Five epochs, two fixed views and float16 prefix caches are declared resource approximations; training converts activations to float32. Feature batches are4, training batches8 and CPU threads2. Resource guards stop on low memory/disk. Training-view caches are removed after each completed model; test prefix activations are retained for verification.''')
source=(ROOT/'baseline.py').read_text().split("if __name__=='__main__':")[0].replace('Path(__file__).resolve().parent','Path.cwd()')
code(source)
code("if MODE == 'reproduce':\n    main()\nelse:\n    print('Five models were trained in the bounded local run. Inspecting saved measured outputs.')\n    assert all((ROOT/'results'/f'{n}.json').exists() for n in MODEL_NAMES)")
md('''## Independent verification of learned classifiers

This recomputes headline metrics, reloads all five learned late blocks/classifiers and predicts from cached frozen test-prefix activations. Predictions must match their saved probabilities. The receipt also reports a618-unique-image sensitivity analysis and file hashes.''')
verify_source=(ROOT/'verify_results.py').read_text().split("if __name__=='__main__':")[0].replace('Path(__file__).resolve().parent','Path.cwd()')
code(verify_source)
code("import torch\ntorch.set_num_threads(2)\nreceipt = verify(require_heads=(MODE == 'reproduce'))\nif not receipt['saved_heads_predictions_verified']:\n    print('Portable package: per-image scores verified. Learned-head verification requires local caches or reproduction mode.')")
md('''## Measured results

These values come from the completed run artifacts, not from paper values or predictions. Ensemble labels use at least three pneumonia votes. ROC AUC uses mean pneumonia probability as the continuous ensemble score.''')
code('''import pandas as pd
summary = json.loads((ROOT/'results/summary.json').read_text())
metrics = ['accuracy','sensitivity','specificity','precision','pneumonia_f1','normal_f1','macro_f1','auc']
rows=[]
for name in MODEL_NAMES:
    r=json.loads((ROOT/'results'/f'{name}.json').read_text())
    rows.append({'model':name,**{k:r['metrics'][k] for k in metrics}})
rows.append({'model':'five_CNN_majority_vote',**{k:summary['majority_vote'][k] for k in metrics}})
display(pd.DataFrame(rows).set_index('model').round(4))
print('Paper target: accuracy .9639, sensitivity .9962, precision .9328, AUC .9934')
print('Our accuracy minus paper, percentage points:',100*(summary['majority_vote']['accuracy']-.9639))
print('Image-level bootstrap intervals, single training seed:',json.dumps(summary['bootstrap_95_ci'],indent=2))
''')
md('''## Confusion matrix and ROC curve

Labels are normal 0 and pneumonia 1. Specificity is normal-class recall. AUC measures score ranking across thresholds. Image-level bootstrap intervals do not include training-seed variability and may be optimistic with repeated or related images.''')
code('''import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
import numpy as np
plt.rcParams.update({'figure.dpi':110,'font.size':11})
manifest=json.loads((ROOT/'data/manifest.json').read_text())
test=[r for r in manifest if r['split']=='test']
y=np.array([r['label'] for r in test])
probs=np.stack([np.load(ROOT/'results'/f'{n}_probabilities.npy') for n in MODEL_NAMES])
pred=(probs>=.5).sum(0)>=3
fig,axes=plt.subplots(1,2,figsize=(11,4))
ConfusionMatrixDisplay(np.asarray(summary['majority_vote']['confusion_matrix']),display_labels=['Normal','Pneumonia']).plot(ax=axes[0],colorbar=False,cmap='Blues')
fpr,tpr,_=roc_curve(y,probs.mean(0))
axes[1].plot(fpr,tpr,label=f"AUC = {summary['majority_vote']['auc']:.3f}")
axes[1].plot([0,1],[0,1],'--',color='gray');axes[1].set(xlabel='False positive rate',ylabel='Pneumonia sensitivity',title='Ensemble ROC');axes[1].legend()
fig.tight_layout();fig.savefig(ROOT/'results/confusion_and_roc.png',bbox_inches='tight');plt.show()
''')
md('''## Actual training histories

Fixed schedules were decided before final testing. The following curves use saved epoch losses. There was no test-score-driven checkpoint selection.''')
code('''fig,ax=plt.subplots(figsize=(9,4))
for n in MODEL_NAMES:
    h=json.loads((ROOT/'results'/f'{n}.json').read_text())['history']
    ax.plot([r['epoch'] for r in h],[r['loss'] for r in h],label=n)
ax.set(xlabel='Epoch',ylabel='Training cross entropy',title='Classifier training histories');ax.legend();fig.tight_layout();plt.show()
''')
md('''## Why accuracy alone is insufficient

An always-pneumonia predictor does not need any image. We compute its test score below. High pneumonia F1 can coexist with zero normal recall. A small dataset or class imbalance does not automatically imply low F1 or overfitting.''')
code("constant = evaluate(y, np.ones(len(y)))\nprint(json.dumps(constant,indent=2))")
md('''## Actual error examples

The examples below are misclassifications by our measured ensemble. They are qualitative inspection, not clinical diagnoses. Wrong predictions alone do not identify their cause.''')
code('''errors=np.flatnonzero(pred!=y)[:6]
from IPython.display import Image as DisplayImage
if not all((ROOT/test[i]['path']).exists() for i in errors):
    display(DisplayImage(filename=str(ROOT/'results/error_examples.png')))
else:
    fig,axes=plt.subplots(2,3,figsize=(10,6))
    for ax,i in zip(axes.flat,errors):
        ax.imshow(Image.open(ROOT/test[i]['path']),cmap='gray')
        ax.set_title(f"True {int(y[i])}, predicted {int(pred[i])}" + chr(10) + f"p(Pneumonia)={probs[:,i].mean():.3f}")
        ax.axis('off')
    for ax in list(axes.flat)[len(errors):]:ax.axis('off')
    fig.tight_layout();fig.savefig(ROOT/'results/error_examples.png',bbox_inches='tight');plt.show()
''')
md((ROOT/'HYPOTHESIS_BEFORE_IMPLEMENTATION.md').read_text())
md('''## Limitations and next milestone

One pediatric hospital and one baseline seed limit generalization. Repeated records remain in the paper-comparison protocol. There is no independent validation subset after merging the original train/val folders. Our reconstruction makes declared assumptions for missing author settings and uses finite cached augmentation. No model was tuned using final test results. The paper-comparison score does not prove exact code reproduction.

M4 will create a fixed group-aware validation protocol and rerun the full-image control under it before comparing the one crop change. The M2 score must not be compared directly to a crop result trained on a different split. Paired bootstrap intervals are planned for metric differences. Heatmaps can illustrate individual CNN activation, but cannot prove lesion localization or clinical correctness. The hypothesis experiment has not been run.

## Sources

- https://doi.org/10.3390/app10020559
- https://eprints.qut.edu.au/199742/1/applsci_10_00559_1_.pdf
- https://huggingface.co/datasets/hf-vision/chest-xray-pneumonia
- https://doi.org/10.17632/rscbjbr9sj.2
- https://github.com/pytorch/vision
- https://pmc.ncbi.nlm.nih.gov/articles/PMC10827725/
- https://pmc.ncbi.nlm.nih.gov/articles/PMC10248870/
''')
book=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3 (CV baseline)','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11'}})
import ast
for cell in cells:
    if cell.cell_type=='code':ast.parse(cell.source)
out=ROOT/'output/M2_Reproduced_Baseline.ipynb'
nb.write(book,out)
import os
os.environ['JUPYTER_PATH']=str(ROOT/'.build/share/jupyter')
client=NotebookClient(book,timeout=900,kernel_name='cvbaseline',resources={'metadata':{'path':str(ROOT)}})
client.execute();nb.write(book,out)
assert not any(o.get('output_type')=='error' for c in book.cells for o in c.get('outputs',[]))
print('Executed notebook saved',out)
