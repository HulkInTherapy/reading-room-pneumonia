"""Lightweight status writer and completion verifier for the current local run."""
import sys,time,json,subprocess
from pathlib import Path
from datetime import datetime
import psutil
HERE=Path(__file__).resolve().parent;OUT=HERE/'results'
pid=int(sys.argv[1]);process=psutil.Process(pid);created=process.create_time()
def update(status):
    models={}
    for name in ['resnet18','densenet121','googlenet','inception_v3','alexnet']:
        f=OUT/f'{name}_history.json';history=json.loads(f.read_text()) if f.exists() else []
        selection=OUT/f'{name}_selection.json'
        models[name]={'completed_epochs':len(history),'target_epochs':15,'training_complete':(OUT/f'{name}_complete.json').exists(),'selected_epoch':json.loads(selection.read_text())['epoch'] if selection.exists() else None}
    receipt=dict(status=status,updated_local=datetime.now().isoformat(timespec='seconds'),models=models,training_pid=pid,results_verified=(OUT/'independent_verification.json').exists(),notebook_exists=(HERE.parents[1]/'output/M2_Audited_15_Epoch_Restart.ipynb').exists())
    (HERE/'STATUS.json').write_text(json.dumps(receipt,indent=2))
while process.is_running() and process.create_time()==created and process.status()!=psutil.STATUS_ZOMBIE:
    update('training');time.sleep(30)
if not (OUT/'summary.json').exists():
    update('training_stopped_before_final_results');sys.exit(1)
update('independent_verification_and_notebook_execution')
with (HERE/'finalization.log').open('w') as log:
    done=subprocess.run([sys.executable,str(HERE/'finalize.py')],stdout=log,stderr=subprocess.STDOUT)
update('complete_and_verified' if done.returncode==0 else 'finalization_failed_check_log')
sys.exit(done.returncode)
