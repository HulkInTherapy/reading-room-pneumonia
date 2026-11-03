"""Independently verify completed experiment and execute its inspection notebook.

Does not train, choose a threshold, or select a checkpoint on the test set.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import sys,json,hashlib,gc
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from baseline import MODEL_NAMES,build_model,evaluate

def verify():
    out=HERE/'results'
    summary=json.loads((out/'summary.json').read_text())
    split=json.loads((HERE/'split.json').read_text())
    rows=split['test_records'];y=np.array([r['label'] for r in rows])
    probabilities=[];checks=[]
    for name in MODEL_NAMES:
        history=json.loads((out/f'{name}_history.json').read_text())
        assert len(history)==15 and [h['epoch'] for h in history]==list(range(1,16))
        selected=max(history,key=lambda h:(h['validation_metrics']['macro_f1'],h['validation_metrics']['auc'],-h['epoch']))
        assert selected['epoch']==summary['selected_epochs'][name]
        prefix,tail,_=build_model(name);prefix.handle.remove();del prefix;gc.collect()
        checkpoint=out/f'{name}_best.pt'
        tail.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));tail.eval()
        x=np.load(ROOT/'cache_tail'/f'{name}_test.npy',mmap_mode='r');p=[]
        with torch.inference_mode():
            for i in range(0,len(x),8):
                z=torch.from_numpy(np.array(x[i:i+8],dtype=np.float32,copy=True))
                p.extend(tail(z).softmax(-1)[:,1].numpy())
        p=np.array(p);stored=np.load(out/f'{name}_test_probabilities.npy')
        difference=float(np.max(np.abs(p-stored)))
        assert np.allclose(p,stored,rtol=1e-4,atol=3e-5),(name,difference)
        checks.append(dict(model=name,selected_epoch=selected['epoch'],epochs_verified=15,
                           max_probability_difference=difference,
                           checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                           test_metrics=evaluate(y,p)))
        probabilities.append(p);del tail,x,z;gc.collect()
    p=np.stack(probabilities);pred=((p>=.5).sum(0)>=3).astype(int)
    metrics=evaluate(y,p.mean(0),pred)
    for key in metrics:
        if key=='confusion_matrix':assert metrics[key]==summary['metrics'][key]
        else:assert abs(metrics[key]-summary['metrics'][key])<1e-8
    receipt=dict(verified=True,models=checks,ensemble=metrics,
                 checkpoint_selection='Independently recomputed from validation histories only',
                 warning=summary['warning'])
    (out/'independent_verification.json').write_text(json.dumps(receipt,indent=2))
    return receipt

def notebook():
    import nbformat as nb
    from nbclient import NotebookClient
    cells=[]
    def md(s):cells.append(nb.v4.new_markdown_cell(s))
    def code(s):cells.append(nb.v4.new_code_cell(s))
    md('''# Pneumonia reproduction audit and 15 epoch restart

Samarth Shekhar 230042 · Meghavi Rao 230044 · Rachit Kumar 230128 · Harshal Nerpagar 230076

Base paper: [Chouhan et al. 2020](https://doi.org/10.3390/app10020559). This notebook executes independent checks and visualizations of the actual completed command-line training run. It does not pretend that training ran inside these notebook cells. All five models were restarted from ImageNet and trained for 15 epochs on this Mac, then selected using validation only. The original five-epoch experiment is preserved. Neither local experiment follows the full 50–200 epoch published schedule. Lung cropping is still an unimplemented M3 hypothesis.''')
    code('''from pathlib import Path
import sys,json,numpy as np,pandas as pd
ROOT=Path('/Users/zebra/cvproj/m2_m3')
EXP=ROOT/'experiments/restart15';OUT=EXP/'results'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(EXP))
from baseline import MODEL_NAMES,evaluate
from finalize import verify
receipt=verify()
print('Independent learned checkpoint verification:',receipt['verified'])
''')
    md((HERE/'AUDIT_AND_PROTOCOL.md').read_text())
    code("print(json.dumps(json.loads((EXP/'split_audit.json').read_text()),indent=2))\nprint(json.dumps(json.loads((EXP/'quantization_audit.json').read_text()),indent=2))")
    md('''## Executable training implementation

Run `nice -n 15 .venv/bin/python -u experiments/restart15/run.py` from m2_m3 for a deliberate training run. Resource guards remain enabled. The source below is displayed rather than automatically launching another multi-hour run. Locally generated resume states include model, optimizer and RNG; completed models are skipped.''')
    code("print((EXP/'run.py').read_text())")
    md('''## Actual test comparison

The same original 624 test records are used for both local results. Published numbers are reported claims, not independently reproduced values. The test set was inspected during earlier work, so it is not a pristine blind holdout. Settings and checkpoints in this restart were not chosen using test scores. Several factors changed together: this is not an isolated epoch-count ablation.''')
    code('''summary=json.loads((OUT/'summary.json').read_text())
keys=['accuracy','precision','sensitivity','specificity','pneumonia_f1','normal_f1','macro_f1','auc']
paper={'accuracy':.9639,'precision':.9328,'sensitivity':.9962,'auc':.9934}
comparison=pd.DataFrame([{'run':'Paper Table 2 reported',**paper},{'run':'Original local 5 epochs',**summary['old_five_epoch_metrics']},{'run':'Fresh 15 epochs validation selected',**summary['metrics']}]).set_index('run')
display(comparison[keys].round(4))
display(pd.DataFrame(receipt['models'])[['model','selected_epoch','epochs_verified','max_probability_difference']])
print('Unique test sensitivity analysis:',summary['unique_test_metrics'])
''')
    md('''## Does training longer help under this restart protocol

Compare checkpoints at 5, 10 and 15 epochs using validation, not test. The selected ensemble combines each model's best validation macro-F1 checkpoint, which need not be the last epoch. Fixed image views cycle every four epochs, so view effects also influence individual epoch losses.''')
    code('''display(pd.DataFrame(summary['validation']['epoch_validation_ensembles']).T[keys].round(4))
print('Selected validation ensemble:',summary['validation']['selected_validation_ensemble'])
import matplotlib.pyplot as plt
fig,axs=plt.subplots(1,2,figsize=(12,4))
for n in MODEL_NAMES:
    h=json.loads((OUT/f'{n}_history.json').read_text())
    axs[0].plot([a['epoch'] for a in h],[a['validation_metrics']['macro_f1'] for a in h],label=n)
    axs[1].plot([a['epoch'] for a in h],[a['validation_loss'] for a in h],label=n)
axs[0].set(xlabel='Epoch',ylabel='Validation macro F1');axs[1].set(xlabel='Epoch',ylabel='Validation cross entropy')
axs[0].legend(fontsize=8);fig.tight_layout();fig.savefig(OUT/'validation_curves.png',dpi=150);plt.show()
''')
    code('''from sklearn.metrics import ConfusionMatrixDisplay,roc_curve
split=json.loads((EXP/'split.json').read_text());y=np.array([r['label'] for r in split['test_records']])
p=np.stack([np.load(OUT/f'{n}_test_probabilities.npy') for n in MODEL_NAMES])
fig,axs=plt.subplots(1,2,figsize=(10,4))
ConfusionMatrixDisplay(np.array(summary['metrics']['confusion_matrix']),display_labels=['Normal','Pneumonia']).plot(ax=axs[0],colorbar=False,cmap='Purples')
fpr,tpr,_=roc_curve(y,p.mean(0));axs[1].plot(fpr,tpr,label=f"AUC {summary['metrics']['auc']:.4f}")
axs[1].plot([0,1],[0,1],'--',color='gray');axs[1].set(xlabel='False positive rate',ylabel='Sensitivity');axs[1].legend();fig.tight_layout();fig.savefig(OUT/'test_confusion_and_roc.png',dpi=150);plt.show()
''')
    md('''## Interpretation and limits

Higher validation than test performance can reflect distribution differences, group proxies or residual leakage, as well as model generalization problems. It does not identify one causal explanation. Cached augmentation, fixed normalization, seed 42, pretrained weight versions, DenseNet's frozen norm5 and unspecified author settings remain differences. One training seed and one hospital do not establish robustness or clinical safety. An ensemble Grad-CAM demo explains a ResNet18 component, not the whole vote or pneumonia lesions.

The revised M3 proposal combines lung cropping with a pretrained small ViT. Cropping remains the primary handout hypothesis, and ViT is a separate extension. Four matched full/crop and CNN-only/CNN-plus-ViT comparisons will separate their effects. Neither improvement has run. Predicted gains remain hypotheses, not measured results. JEPA and mixup remain optional later work. We make no state-of-the-art claim.''')
    md((ROOT/'HYPOTHESIS_BEFORE_IMPLEMENTATION.md').read_text())
    nbk=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'CV Python','language':'python','name':'python3'},'language_info':{'name':'python'}})
    destination=ROOT/'output/M2_Audited_15_Epoch_Restart.ipynb'
    NotebookClient(nbk,timeout=900,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}}).execute()
    nb.write(nbk,destination)
    return destination

def report():
    out=HERE/'results';s=json.loads((out/'summary.json').read_text())
    r=json.loads((out/'independent_verification.json').read_text());assert r['verified']
    import html
    rows=[]
    published={'accuracy':.9639,'precision':.9328,'sensitivity':.9962,'auc':.9934}
    for label,m in [('Paper Table 2',published),('Original 5 epochs',s['old_five_epoch_metrics']),('Restart 15 epochs, validation selected',s['metrics'])]:
        rows.append('<tr><th>'+html.escape(label)+'</th>'+''.join('<td>'+('—' if key not in m else f'{100*m[key]:.2f}%')+'</td>' for key in ['accuracy','precision','sensitivity','specificity','macro_f1','auc'])+'</tr>')
    epochrows=[]
    for e,m in s['validation']['epoch_validation_ensembles'].items():
        epochrows.append(f'<tr><th>{e}</th><td>{m["macro_f1"]:.4f}</td><td>{m["auc"]:.4f}</td><td>{100*m["specificity"]:.2f}%</td></tr>')
    content='''<!doctype html><html lang="en"><meta charset="utf-8"><title>Pneumonia reproduction audit</title>
<style>body{font:17px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;max-width:1040px;margin:60px auto;padding:0 28px;color:#20212b;background:#faf9f6}h1{font-size:38px;line-height:1.15;letter-spacing:-1px}h2{margin-top:40px;font-size:25px}table{width:100%;border-collapse:collapse;background:white;font-size:15px}th,td{padding:14px;border-bottom:1px solid #ddd;text-align:right}th:first-child{text-align:left}thead{background:#eae5fa}a{color:#5841ab}img{width:100%;height:auto}.muted{color:#62606c}</style>
<h1>Pneumonia reproduction audit</h1><p>All five CNNs restarted from ImageNet and completed 15 epochs. Checkpoints were selected by validation macro F1, not by the test set. Learned weights and saved probabilities passed independent reload verification.</p>
<h2>Published and actual results</h2><table><thead><tr><th>Run</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>Specificity</th><th>Macro F1</th><th>AUC</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table>
<p class="muted">Published numbers are reported claims. The paper's precision, recall and accuracy do not reconcile to one confusion matrix on its stated 624 images; averages or an undocumented evaluation definition could explain this. This is not proof of fabrication.</p>
<h2>What was different</h2><p>The old run used five epochs, two cached views, batch eight, the general LR schedule for AlexNet and no independent checkpoint-selection set. The restart uses deduplicated group-separated training/validation, batch sixteen, four cyclic views including explicitly assumed Gaussian noise, AlexNet's stated high-then-low step schedule and validation-selected checkpoints. These changes are bundled: a gain cannot be assigned to epochs alone.</p>
<p>Table 2 reports 200 epochs for ResNet18/AlexNet, 100 for DenseNet121/InceptionV3 and 50 for GoogLeNet. AlexNet prose says 200 iterations before lowering LR; the relationship to its table epoch count is ambiguous. Fifteen local epochs is still a truncated reproduction.</p>
<h2>Do more epochs help</h2><table><thead><tr><th>Epoch checkpoint</th><th>Validation macro F1</th><th>Validation AUC</th><th>Validation specificity</th></tr></thead><tbody>'''+''.join(epochrows)+'''</tbody></table><p>Selected epochs: '''+html.escape(str(s['selected_epochs']))+'''. Do not assume the final epoch is best. Test scores were not used to choose these epochs.</p>
<h2>Checks and unresolved limits</h2><p>No exact source hash or recoverable filename group crosses training/validation. The strict perceptual screen found no near-duplicate candidate across train/validation or non-test/test; this does not prove patient independence. On 16 balanced non-test images per model, float16 conversion changed no decisions and probabilities by at most 0.000055. It is unlikely to explain a ten-point gap by itself.</p>
<p>Training prevalence is about 74.2% pneumonia versus 62.5% in test. High AUC with poor normal recall suggests operating-point and generalization issues, but neither prevalence nor these audits establishes a single causal explanation. The original benchmark test was previously inspected and is not a blind external holdout. Frozen caches, fixed augmentation, seed choice, normalization, classifier details and DenseNet norm5 remain assumptions; author code was not verified.</p>
<h2>Actual curves and errors</h2><img src="validation_curves.png" alt="Validation curves across all fifteen epochs"><img src="test_confusion_and_roc.png" alt="Final test confusion matrix and ROC curve">
<p>The M3 proposal now combines lung cropping and a pretrained small ViT, with four matched comparisons to isolate each effect. Neither improvement has run. JEPA and mixup remain optional future work. No clinical or state-of-the-art claim is made.</p><p><a href="https://doi.org/10.3390/app10020559">Chouhan et al. 2020</a> · <a href="../AUDIT_AND_PROTOCOL.md">Full protocol</a> · <a href="independent_verification.json">Verification receipt</a></p></html>'''
    path=out/'AUDIT_RESULTS.html';path.write_text(content);return path

if __name__=='__main__':
    print(json.dumps(verify(),indent=2),flush=True)
    print('EXECUTED_NOTEBOOK',notebook(),flush=True)
    print('AUDIT_REPORT',report(),flush=True)
