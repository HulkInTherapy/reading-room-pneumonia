"""Tavily search and direct primary-source retrieval; secrets never printed."""
import json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
def search(query, name):
    keys = [l.split('=',1)[1].strip() for l in (ROOT.parent/'.env').read_text().splitlines() if l.startswith('TAVILY_API_KEY')]
    for key in keys:
        body = json.dumps(dict(api_key=key, query=query, search_depth='advanced', max_results=5))
        p = subprocess.run(['curl','-sS','--max-time','45','-X','POST','https://api.tavily.com/search','-H','Content-Type: application/json','-d',body], capture_output=True,text=True)
        try: data=json.loads(p.stdout)
        except ValueError: continue
        if 'results' in data:
            (ROOT/'sources').mkdir(exist_ok=True)
            (ROOT/'sources'/f'{name}.json').write_text(json.dumps(data,indent=2))
            for r in data['results']: print(r['title'],r['url'],r['content'][:700],sep='\n')
            return
    raise RuntimeError('All Tavily keys failed; no research fabricated')
if __name__=='__main__': search(sys.argv[1],sys.argv[2])
