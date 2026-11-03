"""Readable PDFs from presenter Markdown sources; no model execution."""
import argparse,html,re
from pathlib import Path
from PIL import Image as PILImage
from reportlab.platypus import SimpleDocTemplate,Paragraph,LongTable,TableStyle,Image,Spacer
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.colors import HexColor
R=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--input',default='Presenter_Notes.md');parser.add_argument('--output');args=parser.parse_args()
guide='Guide' in args.input;destination=args.output or Path(args.input).with_suffix('.pdf').name
styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='BodyRR',fontName='Helvetica',fontSize=11 if guide else 10,leading=15.5 if guide else 13,spaceAfter=8 if guide else 6,allowWidows=0,allowOrphans=0));styles.add(ParagraphStyle(name='HeadRR',fontName='Helvetica-Bold',fontSize=15 if guide else 14,leading=19,textColor=HexColor('#087f8c'),spaceBefore=13,spaceAfter=8,keepWithNext=True));styles.add(ParagraphStyle(name='TitleRR',fontName='Helvetica-Bold',fontSize=23,leading=28,textColor=HexColor('#132736'),spaceAfter=14));styles.add(ParagraphStyle(name='TableRR',fontName='Helvetica',fontSize=10,leading=13,spaceAfter=2));styles.add(ParagraphStyle(name='QuoteRR',fontName='Helvetica-Oblique',fontSize=11,leading=16,spaceBefore=6,spaceAfter=12,textColor=HexColor('#132736'),backColor=HexColor('#eaf5f5'),borderPadding=8));story=[];rows=[]
def para(s,style='BodyRR'):
    s=html.escape(s).replace('`','');s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s)
    return Paragraph(s,styles[style])
def flush():
    if not rows:return
    n=len(rows[0]);widths={2:[154,358],3:[70,302,140],4:[42,120,260,90]}.get(n,[512/n]*n)
    t=LongTable(rows,colWidths=widths,repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor('#eaf5f5')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOX',(0,0),(-1,-1),.4,HexColor('#cedbdd')),('INNERGRID',(0,0),(-1,-1),.3,HexColor('#dce5e5')),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    story.append(t);rows.clear()
for raw in (R/args.input).read_text().splitlines():
    line=raw.strip()
    if line.startswith('|'):
        parts=[x.strip() for x in line.strip('|').split('|')]
        if all(set(x)<=set('-: ') for x in parts):continue
        rows.append([para(x,'TableRR') for x in parts]);continue
    flush()
    if not line:continue
    image=re.fullmatch(r'!\[(.*?)\]\((.*?)\)',line)
    if image:
        path=R/image[2]
        with PILImage.open(path) as im:w,h=im.size
        width=512;story.append(Image(str(path),width=width,height=width*h/w));story.append(Spacer(1,10));continue
    if line.startswith('# '):story.append(para(line[2:],'TitleRR'))
    elif line.startswith('## '):story.append(para(line[3:],'HeadRR'))
    elif line.startswith('> '):story.append(para(line[2:],'QuoteRR'))
    else:story.append(para(line))
flush()
def footer(c,doc):
    c.setFont('Helvetica',8);c.setFillColor(HexColor('#637586'));c.drawString(42,22,'Reading Room / '+('20-minute understanding guide' if guide else 'presenter copy'));c.drawRightString(570,22,str(doc.page))
SimpleDocTemplate(str(R/destination),pagesize=(612,792),leftMargin=40 if guide else 36,rightMargin=40 if guide else 36,topMargin=36,bottomMargin=34).build(story,onFirstPage=footer,onLaterPages=footer)
print('Built',destination)
