from pathlib import Path
import json
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;O=R/'results'
assert json.loads((O/'verification.json').read_text())['verified']
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
for file in sorted(O.glob('*_history.json')):
 h=json.loads(file.read_text());name=file.stem.replace('_history','').replace('_',' ')
 axes[0].plot([r['epoch'] for r in h],[r['loss'] for r in h],marker='o',label=name)
 axes[1].plot([r['epoch'] for r in h],[r['validation']['macro_f1'] for r in h],marker='o',label=name)
for ax in axes:ax.set_xlabel('Head-training epoch');ax.set_xticks([1,2,3,4,5]);ax.grid(alpha=.2);ax.legend(fontsize=8)
axes[0].set_ylabel('Training cross-entropy');axes[1].set_ylabel('Validation macro F1');axes[1].set_ylim(top=1.005)
fig.suptitle('Frozen encoders, five classifier-head epochs; validation selects checkpoints')
fig.tight_layout();fig.savefig(O/'training_histories.png',dpi=160);plt.close(fig)
