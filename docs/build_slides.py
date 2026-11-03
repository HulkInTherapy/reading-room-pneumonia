"""Eight visual presenter slides, rebuilt from measured public project artifacts."""
import argparse,base64,html,json,io,textwrap
from pathlib import Path
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
import pymupdf,qrcode
P=Path(__file__).resolve().parent;ROOT=P.parent
C={'bg':'#F5F4EF','ink':'#132736','teal':'#087F8C','violet':'#7560CA','coral':'#D46451','muted':'#637586','line':'#D9E2E6'}
O=P/'Presentation_8_Slides.pdf';c=canvas.Canvas(str(O),pagesize=(1280,720));c.setTitle('Reading Room — Eight-slide final presentation')
def text(s,x,y,size=28,color='ink',bold=False):
 c.setFillColor(HexColor(C.get(color,color)));c.setFont('Helvetica-Bold' if bold else 'Helvetica',size);c.drawString(x,y,s)
def start(title,n,dark=False):
 c.setFillColor(HexColor(C['ink'] if dark else C['bg']));c.rect(0,0,1280,720,fill=1,stroke=0)
 if title:text(title,64,638,43,'ink',True)
 text('Reading Room / final project',64,26,14,'#A8C4CC' if dark else 'muted');text(f'{n} / 8',1160,26,14,'#A8C4CC' if dark else 'muted')
def img(path,x,y,w,h):c.drawImage(str(path),x,y,width=w,height=h,preserveAspectRatio=True,anchor='c',mask='auto')
def line(x1,y1,x2,y2,color='line',width=2):c.setStrokeColor(HexColor(C[color]));c.setLineWidth(width);c.line(x1,y1,x2,y2)
S=ROOT/'samples'
start('',1,True);text('READING',64,523,72,'#FFFFFF',True);text('ROOM',64,439,72,'#FFFFFF',True);text('Pneumonia from chest X-rays',68,365,30,'#C1D7DE');text('Samarth Shekhar     Meghavi Rao',68,173,22,'#FFFFFF');text('Rachit Kumar          Harshal Nerpagar',68,136,22,'#FFFFFF');text('Computer Vision / M1-M5',68,82,18,'#A8C4CC');img(S/'117/full.png',825,193,350,350);c.showPage()
start('Two dataset labels',2);img(S/'233/full.png',207,211,330,330);img(S/'429/full.png',745,211,330,330);text('Normal',319,147,32,'teal',True);text('Pneumonia',821,147,32,'coral',True);text('We predict a benchmark label.',408,80,24,'muted');c.showPage()
start('Our project journey',3)
for i,(m,label) in enumerate([('M1','Review'),('M2','Baseline'),('M3','Hypothesis'),('M4','Experiment'),('M5','Demo + analysis')]):
 x=132+i*250
 if i<4:line(x+42,420,x+208,420)
 c.setFillColor(HexColor(C['teal'] if i==4 else C['ink']));c.circle(x,420,39,fill=1,stroke=0);text(m,x-26,410,26,'#FFFFFF',True);c.setFillColor(HexColor(C['ink']));c.setFont('Helvetica-Bold',23);c.drawCentredString(x,339,label)
for i,(label,value) in enumerate([('Cached 5',87.18),('Audit 15',85.74),('Online 5',92.15),('Final pilot',93.75)]):
 x=118+i*292;text(f'{value:.2f}%',x,189,38,'teal' if i==3 else 'muted',True);text(label,x+10,145,20,'muted')
text('Historical runs used different settings.',395,82,19,'muted');c.showPage()
start('Lung-focused input',4);img(P/'figures/segmentation_pipeline.png',64,229,1152,365)
text('CNN',360,148,32,'teal',True);text('+',546,148,34,'muted');text('Transformer',636,148,32,'violet',True);text('Two pretrained image readers',414,88,23,'muted');c.showPage()
start('Four matched comparisons',5);text('Full image',365,558,27,'ink',True);text('Lung crop',895,558,27,'ink',True);text('CNN',64,427,24,'muted',True);text('CNN + ViT',64,218,24,'muted',True)
line(218,320,1210,320);line(716,123,716,536)
for k,value,x,y,desc in [('A',86.86,310,401,'CNN'),('B',92.63,838,401,'CNN'),('C',88.14,310,194,'CNN + ViT'),('D',93.75,838,194,'CNN + ViT')]:
 text(k,x,y+61,24,'muted',True);text(f'{value:.2f}%',x+40,y,59,'teal' if k=='D' else 'ink',True)
text('Same split. Five classifier-head epochs.',376,72,22,'muted');c.showPage()
start('Final model results',6);text('93.75%',94,424,98,'teal',True);text('Accuracy',101,363,31,'ink',True);text('585 / 624 images correct',101,312,25,'muted');text('98.21%',724,424,98,'violet',True);text('Pneumonia recall',729,363,31,'ink',True);text('383 / 390 pneumonia cases found',729,312,25,'muted');line(64,259,1216,259);text('32',263,155,57,'coral',True);text('false alarms',337,166,27,'muted');text('7',789,155,57,'coral',True);text('missed cases',838,166,27,'muted');c.showPage()
start('Mistakes and limits',7)
# The report's eight-case figure preserves the same deterministic error selection.
# Individual cropped examples are copied as slide assets during publication preparation.
img(P/'figures/slide_false_positive.png',235,232,252,252);img(P/'figures/slide_false_negative.png',795,232,252,252)
text('False alarm',282,179,29,'coral',True);text('Missed pneumonia',797,179,29,'coral',True);text('Validation and test disagreed.',390,86,27,'muted');c.showPage()
start('Live demonstration',8);img(ROOT/'web/assets/demo_desktop.png',60,123,820,475)
for i,t in enumerate(['Choose an image','Click Analyze','Show mask + result']):text(f'{i+1}. {t}',904,533-60*i,24,'ink',True)
qr=qrcode.make('https://hulkintherapy.github.io/reading-room-pneumonia/');buf=io.BytesIO();qr.save(buf,format='PNG');c.drawImage(ImageReader(buf),944,128,width=180,height=180);text('Code, model and guide',923,94,19,'muted');text('hulkintherapy.github.io/reading-room-pneumonia/',64,60,18,'muted');c.showPage();c.save()
# Self-contained image slideshow: identical layout to the checked PDF, zero runtime dependencies.
d=pymupdf.open(O);slides=[];images=P/'slide_images';images.mkdir(exist_ok=True)
for i,page in enumerate(d):
 pix=page.get_pixmap(alpha=False);raw=pix.tobytes('png');(images/f'slide_{i+1:02}.png').write_bytes(raw);slides.append('<section><img alt="Presentation slide '+str(i+1)+'" src="data:image/png;base64,'+base64.b64encode(raw).decode()+'"></section>')
(P/'Presentation_8_Slides.html').write_text('''<!doctype html><meta charset="utf-8"><title>Reading Room / 8 slides</title><style>html,body{margin:0;background:#132736;height:100%;font:14px Arial}section{display:none;height:100vh;align-items:center;justify-content:center}section.on{display:flex}img{max-width:100vw;max-height:100vh;object-fit:contain}nav{position:fixed;bottom:12px;right:14px;background:#ffffffdd;border-radius:9px;padding:6px}button{padding:7px 12px;border:0;background:transparent;cursor:pointer}</style>'''+''.join(slides)+'''<nav><button onclick="go(-1)">Previous</button><span id="counter"></span><button onclick="go(1)">Next</button><button onclick="document.documentElement.requestFullscreen()">Fullscreen</button></nav><script>let s=[...document.querySelectorAll('section')],i=0;function go(n){i=Math.max(0,Math.min(s.length-1,i+n));s.forEach((x,j)=>x.classList.toggle('on',i===j));document.getElementById('counter').textContent=(i+1)+' / '+s.length}document.onkeydown=e=>{if(['ArrowRight',' ','PageDown'].includes(e.key)){e.preventDefault();go(1)}if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();go(-1)}};go(0)</script>''')
print('Built exactly',len(d),'slides')
