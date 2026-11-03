"""Print-friendly presenter notes from the checked Markdown source."""
from pathlib import Path
import html,re
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,LongTable,TableStyle,KeepTogether
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.colors import HexColor
R=Path(__file__).resolve().parent;styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='BodyRR',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=7));styles.add(ParagraphStyle(name='HeadRR',fontName='Helvetica-Bold',fontSize=14,leading=18,textColor=HexColor('#087f8c'),spaceBefore=12,spaceAfter=8,keepWithNext=True));styles.add(ParagraphStyle(name='TitleRR',fontName='Helvetica-Bold',fontSize=23,leading=28,textColor=HexColor('#132736'),spaceAfter=14));story=[];rows=[]
def para(s,style='BodyRR'):
 s=html.escape(s);s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s);s=s.replace('`','');return Paragraph(s,styles[style])
def flush():
 if rows:
  t=LongTable(rows,colWidths=[154,358],repeatRows=1);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor('#eaf5f5')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOX',(0,0),(-1,-1),.4,HexColor('#cedbdd')),('INNERGRID',(0,0),(-1,-1),.3,HexColor('#dce5e5')),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));story.append(t);rows.clear()
for line in (R/'Presenter_Notes.md').read_text().splitlines():
 if line.startswith('|'):
  parts=[x.strip() for x in line.strip('|').split('|')]
  if all(set(x)<=set('-: ') for x in parts):continue
  rows.append([para(x) for x in parts]);continue
 flush()
 if not line.strip():continue
 if line.startswith('# '):story.append(para(line[2:],'TitleRR'))
 elif line.startswith('## '):story.append(para(line[3:],'HeadRR'))
 else:story.append(para(line))
flush()
def footer(c,doc):
 c.setFont('Helvetica',8);c.setFillColor(HexColor('#637586'));c.drawString(42,22,'Reading Room / presenter copy');c.drawRightString(570,22,str(doc.page))
SimpleDocTemplate(str(R/'Presenter_Notes.pdf'),pagesize=(612,792),leftMargin=42,rightMargin=42,topMargin=40,bottomMargin=38).build(story,onFirstPage=footer,onLaterPages=footer)
