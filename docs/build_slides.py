"""Eight visual slides with enough explanation to present without code knowledge."""
import base64
import io
from pathlib import Path
import pymupdf
import qrcode
from reportlab.pdfgen import canvas
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
P=Path(__file__).resolve().parent
ROOT=P.parent
C={'bg':'#F5F4EF','ink':'#132736','teal':'#087F8C','violet':'#7560CA','coral':'#D46451','muted':'#637586','line':'#D9E2E6','light':'#E7F2F2'}
O=P/'Presentation_8_Slides.pdf'
c=canvas.Canvas(str(O),pagesize=(1280,720))
c.setTitle('Reading Room - explanatory eight-slide final presentation')
def text(value,x,y,size=25,color='ink',bold=False):
    c.setFillColor(HexColor(C.get(color,color)))
    c.setFont('Helvetica-Bold' if bold else 'Helvetica',size)
    c.drawString(x,y,value)
def para(value,x,y,width,size=21,leading=27,color='muted',bold=False):
    font='Helvetica-Bold' if bold else 'Helvetica'
    lines=[];line=''
    for word in value.split():
        trial=(line+' '+word).strip()
        if line and stringWidth(trial,font,size)>width:
            lines.append(line);line=word
        else:line=trial
    if line:lines.append(line)
    for line in lines:
        text(line,x,y,size,color,bold);y-=leading
    return y

def start(title,n,dark=False):
    c.setFillColor(HexColor(C['ink'] if dark else C['bg']))
    c.rect(0,0,1280,720,fill=1,stroke=0)
    if title:text(title,64,638,43,'ink',True)
    foot='#A8C4CC' if dark else 'muted'
    text('Reading Room / final project',64,26,14,foot)
    text(f'{n} / 8',1160,26,14,foot)
def img(path,x,y,width,height):
    c.drawImage(str(path),x,y,width=width,height=height,preserveAspectRatio=True,anchor='c',mask='auto')
def line(x1,y1,x2,y2,color='line',width=2):
    c.setStrokeColor(HexColor(C[color]));c.setLineWidth(width);c.line(x1,y1,x2,y2)
S=ROOT/'samples'

start('',1,True)
text('READING',64,523,72,'#FFFFFF',True);text('ROOM',64,439,72,'#FFFFFF',True)
text('Pneumonia from chest X-rays',68,365,30,'#C1D7DE')
para('We test whether focusing on the lungs improves normal-versus-pneumonia classification.',68,310,650,24,32,'#D7E7ED')
text('Samarth Shekhar     Meghavi Rao',68,173,22,'#FFFFFF')
text('Rachit Kumar          Harshal Nerpagar',68,136,22,'#FFFFFF')
text('Computer Vision / M1-M5',68,82,18,'#A8C4CC')
img(S/'117/full.png',825,193,350,350);c.showPage()

start('Dataset and prediction task',2)
img(S/'233/full.png',104,300,235,235);img(S/'429/full.png',382,300,235,235)
text('Normal',161,253,29,'teal',True);text('Pneumonia',427,253,29,'coral',True)
text('Public Kermany dataset (2018)',690,522,29,'ink',True)
para('Pediatric chest X-rays with two labels: Normal and Pneumonia.',690,478,518,23,30)
text('5,856 original image records',690,394,28,'teal',True)
para('Final split: 4,164 training / 1,042 validation / 624 test records.',690,348,518,22,29,'ink')
text('Test: 234 Normal + 390 Pneumonia',690,263,22,'muted')
text("Source: Kermany et al., Guangzhou Women and Children's Medical Center.",64,178,22,'ink')
text('Original split: 5,216 training / 16 validation / 624 test records.',64,133,21,'muted')
text('One pediatric source center; the benchmark is not an external clinical evaluation.',64,90,21,'muted')
c.showPage()

start('Our journey: M1 to M5',3)
text('We kept the failed attempts and audited them before improving the system.',64,581,24,'muted')
xs=[82,230,1050]
c.setFillColor(HexColor(C['light']));c.rect(64,515,1152,43,fill=1,stroke=0)
for x,label in zip(xs,['Stage','What we did','Result']):text(label,x,529,22,'ink',True)
rows=[('M1','Literature review and initial lung-focused idea','Proposal'),('M2','First cached CNN training (5 epochs)','87.18%'),('M2','Longer cached audit (15 epochs)','85.74%'),('M2','Corrected online processing + fresh augmentation','92.15%'),('M3','Recorded crop + transformer hypothesis','Prediction'),('M4','Four matched comparisons; final crop CNN-ViT','93.75%'),('M5','Errors, verification, report and working demo','Verified')]
for i,(stage,work,result) in enumerate(rows):
    y=477-i*47
    text(stage,xs[0],y,23,'ink',True);text(work,xs[1],y,22,'ink')
    text(result,xs[2],y,25,'teal' if i==5 else 'muted',i==5)
    line(64,y-15,1216,y-15,width=.7)
para('Historical runs changed several settings. Only A/B/C/D provide the matched comparison of cropping and the extra model.',64,109,1152,20,25)
c.showPage()

start('From the full X-ray to a lung crop',4)
text('Crop the original pixels first; resize only after the lung-focused region is chosen.',64,581,23,'muted')
img(P/'figures/segmentation_pipeline.png',64,263,1152,300)
for x,title,body in [
    (64,'1 / Full image','Start with the complete X-ray, including lungs and surrounding anatomy.'),
    (455,'2 / Predicted mask','Fixed PSPNet estimates both lungs. White means predicted lung pixels, not disease.'),
    (846,'3 / Lung crop RGB','Keep a rectangle plus 10% margin. Resize to 224 x 224 in three-channel RGB format.')]:
    text(title,x,217,23,'teal',True);para(body,x,181,355,20,25)
text('ResNet18 + DeiT-Tiny classify the crop; final D combines their scores with equal weights.',64,62,21,'ink')
c.showPage()

start('Four matched comparisons',5)
text('Same split and training; vary the input region and whether the transformer is added.',64,582,23,'muted')
text('Full image',365,528,27,'ink',True);text('Lung crop',895,528,27,'ink',True)
text('CNN',64,415,24,'muted',True);text('CNN + ViT',64,204,24,'muted',True)
line(218,313,1210,313);line(716,125,716,510)
for k,value,x,y,desc in [
    ('A',86.86,310,405,'Full X-ray, CNN only: the matched control.'),
    ('B',92.63,838,405,'Lung crop, same CNN: B versus A tests cropping.'),
    ('C',88.14,310,195,'Full X-ray plus transformer: C versus A tests the extra model.'),
    ('D',93.75,838,195,'Crop plus both models: D versus B tests ViT on crops.')]:
    text(k,x,y+58,23,'muted',True);text(f'{value:.2f}%',x+40,y,55,'teal' if k=='D' else 'ink',True)
    para(desc,270 if k in 'AC' else 805,y-44,405,20,24)
text('CNN = ResNet18; ViT = DeiT-Tiny. Four heads trained five epochs; encoders stayed fixed.',64,62,20,'muted')
c.showPage()

start('Final model results',6)
text('Condition D: lung crop + CNN/ViT, evaluated on 624 benchmark test records.',64,582,24,'muted')
text('93.75%',94,440,88,'teal',True);text('Accuracy',101,378,31,'ink',True)
text('585 / 624 images correct',101,329,25,'muted')
text('98.21%',724,440,88,'violet',True);text('Pneumonia recall',729,378,31,'ink',True)
text('383 / 390 pneumonia cases found',729,329,25,'muted')
line(64,279,1216,279)
text('32',146,198,54,'coral',True);text('false alarms',223,209,27,'ink')
para('Normal-labelled images predicted as pneumonia.',101,158,490,21,26)
text('7',773,198,54,'coral',True);text('missed cases',820,209,27,'ink')
para('Pneumonia-labelled images predicted as normal.',729,158,490,21,26)
text('Against the matched full-image CNN, accuracy increased by 6.89 percentage points.',64,67,22,'muted')
c.showPage()

start('Mistakes, limits and next steps',7)
text('These are actual wrong predictions; a good benchmark score does not remove errors.',64,581,23,'muted')
img(P/'figures/slide_false_positive.png',186,303,240,240);img(P/'figures/slide_false_negative.png',824,303,240,240)
text('False alarm',186,264,27,'coral',True);text('Missed pneumonia',824,264,27,'coral',True)
para('A normal-labelled image was predicted as pneumonia.',96,224,520,21,26)
para('A pneumonia-labelled image was predicted as normal.',700,224,516,21,26)
line(64,168,1216,168)
text('Limit: cropping helped test results but reduced performance on this validation split.',64,133,22,'ink')
para('Next: repeat group-aware splits, test independent data and evaluate limited fine-tuning.',64,94,1152,21,26)
text('Poor masks use a full-image fallback (82 / 5,824 unique images). Masks do not mark disease.',64,53,18,'muted')
c.showPage()

start('Live demonstration and this term\'s work',8)
text('We delivered a measured crop/ViT comparison, verified model, web app and GitHub package.',64,581,23,'muted')
img(P/'figures/demo_workspace.png',64,184,778,365)
for i,t in enumerate(['Choose an image','Click Analyze','Show mask + result']):text(f'{i+1}. {t}',904,521-58*i,24,'ink',True)
para('Built-in masks are prepared; classifiers run now. Uploads run segmentation too.',904,319,304,20,25)
qr=qrcode.make('https://hulkintherapy.github.io/reading-room-pneumonia/');buf=io.BytesIO();qr.save(buf,format='PNG')
c.drawImage(ImageReader(buf),953,104,width=160,height=160)
text('Slides, guide, code and model',915,77,18,'muted')
para('Backup: open the recorded offline demo and explain that it contains saved actual model responses.',64,144,764,20,26)
text('hulkintherapy.github.io/reading-room-pneumonia/',64,54,18,'teal')
c.showPage();c.save()

# The HTML uses the exact checked PDF rendering and works without a server.
d=pymupdf.open(O);parts=[];images=P/'slide_images';images.mkdir(exist_ok=True)
for i,page in enumerate(d):
    raw=page.get_pixmap(alpha=False).tobytes('png');(images/f'slide_{i+1:02}.png').write_bytes(raw)
    parts.append('<section><img alt="Presentation slide '+str(i+1)+'" src="data:image/png;base64,'+base64.b64encode(raw).decode()+'"></section>')
(P/'Presentation_8_Slides.html').write_text('''<!doctype html><meta charset="utf-8"><title>Reading Room / 8 slides</title><style>html,body{margin:0;background:#132736;height:100%;font:14px Arial}section{display:none;height:100vh;align-items:center;justify-content:center}section.on{display:flex}img{max-width:100vw;max-height:100vh;object-fit:contain}nav{position:fixed;bottom:12px;right:14px;background:#ffffffdd;border-radius:9px;padding:6px}button{padding:7px 12px;border:0;background:transparent;cursor:pointer}</style>'''+''.join(parts)+'''<nav><button onclick="go(-1)">Previous</button><span id="counter"></span><button onclick="go(1)">Next</button><button onclick="document.documentElement.requestFullscreen()">Fullscreen</button></nav><script>let s=[...document.querySelectorAll('section')],i=0;function go(n){i=Math.max(0,Math.min(s.length-1,i+n));s.forEach((x,j)=>x.classList.toggle('on',i===j));document.getElementById('counter').textContent=(i+1)+' / '+s.length}document.onkeydown=e=>{if(['ArrowRight',' ','PageDown'].includes(e.key)){e.preventDefault();go(1)}if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();go(-1)}};go(0)</script>''')
assert len(d)==8
print('Built eight explanatory slides')
