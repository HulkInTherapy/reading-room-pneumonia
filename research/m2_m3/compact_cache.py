"""Compact only images downloaded by this task; retain source SHA manifests."""
import json
from pathlib import Path
from PIL import Image
root=Path(__file__).resolve().parent
for f in (root/'data').glob('*.manifest.json'):
    rows=json.loads(f.read_text())
    for r in rows:
        old=root/r['path'];new=old.with_suffix('.png')
        if old.suffix=='.jpg':
            if old.exists():
                with Image.open(old) as im:im.convert('RGB').resize((224,224),Image.Resampling.BILINEAR).save(new)
                old.unlink()
            if not new.exists():raise RuntimeError(f'Missing source and resized file: {old}')
            r['path']=str(new.relative_to(root))
    f.write_text(json.dumps(rows,indent=2))
print('Compacted own downloaded image caches; source SHA256 retained.')
