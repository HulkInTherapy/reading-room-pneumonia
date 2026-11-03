"""Open the prepared local demo. No retraining and no external hosting."""
import json,subprocess,sys,time,urllib.request,webbrowser
from pathlib import Path
ROOT=Path(__file__).resolve().parent;URL='http://127.0.0.1:8765'
def running():
    try:
        with urllib.request.urlopen(URL+'/api/status',timeout=2) as r:return json.load(r).get('model')=='Five CNNs, majority vote'
    except Exception:return False
if running():webbrowser.open(URL);print('Demo is already running at',URL);sys.exit(0)
python=ROOT/'.venv/bin/python'
if not python.exists():raise SystemExit('Use the prepared Mac folder or install the requirements first.')
p=subprocess.Popen([str(python),'-m','uvicorn','demo.app:app','--host','127.0.0.1','--port','8765'],cwd=ROOT)
try:
    for _ in range(45):
        if running():break
        if p.poll() is not None:raise SystemExit('Demo server could not start.')
        time.sleep(1)
    if not running():raise SystemExit('Demo startup timed out.')
    webbrowser.open(URL);print('Keep this terminal open. Press Ctrl+C to stop.')
    p.wait()
except KeyboardInterrupt:p.terminate();p.wait()
