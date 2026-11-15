#!/usr/bin/env python3
"""Render the editable proposal source to DOCX and PDF. No inference is run."""
import argparse
from html import escape
from pathlib import Path
import re
import textwrap

from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image as PDFImage, XPreformatted

HERE = Path(__file__).resolve().parent
FONT_ROOT = Path('/System/Library/Fonts/Supplemental')


def font_path(name):
    choices = {
        'regular': [FONT_ROOT/'Arial.ttf', Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')],
        'bold': [FONT_ROOT/'Arial Bold.ttf', Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')],
        'italic': [FONT_ROOT/'Arial Italic.ttf', Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf')],
        'mono': [FONT_ROOT/'Courier New.ttf', Path('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf')],
    }
    return next((p for p in choices[name] if p.exists()), None)


def figure():
    width, height = 1760, 460
    im = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(im)
    regular = ImageFont.truetype(str(font_path('regular')), 28)
    bold = ImageFont.truetype(str(font_path('bold')), 29)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">', '<rect width="100%" height="100%" fill="white"/>']

    def box(x, y, w, h, title, lines, fill='#F1F4F7'):
        draw.rounded_rectangle((x,y,x+w,y+h), radius=10, fill=fill, outline='#7A8793', width=2)
        svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="#7A8793" stroke-width="2"/>')
        for i,line in enumerate([title,*lines]):
            f = bold if i == 0 else regular
            yy = y+18+i*37
            draw.text((x+w/2,yy),line,font=f,fill='black',anchor='mt')
            svg.append(f'<text x="{x+w/2}" y="{yy+28}" text-anchor="middle" font-family="Arial" font-size="{29 if i==0 else 28}" font-weight="{"bold" if i==0 else "normal"}">{escape(line)}</text>')

    def arrow(points):
        draw.line(points, fill='#465461', width=4)
        x,y=points[-1];px,py=points[-2]
        if x>px: tri=[(x,y),(x-14,y-8),(x-14,y+8)]
        elif x<px:tri=[(x,y),(x+14,y-8),(x+14,y+8)]
        elif y>py:tri=[(x,y),(x-8,y-14),(x+8,y-14)]
        else:tri=[(x,y),(x-8,y+14),(x+8,y+14)]
        draw.polygon(tri,fill='#465461')
        svg.append('<polyline points="'+' '.join(f'{a},{b}' for a,b in points)+'" fill="none" stroke="#465461" stroke-width="4"/>')
        svg.append('<polygon points="'+' '.join(f'{a},{b}' for a,b in tri)+'" fill="#465461"/>')

    box(10,125,275,150,'GGUF and tokens',['Model contract','Packed weights'])
    box(335,125,310,150,'llama.cpp graph',['Layer 0 and others','Backend allocation'])
    box(700,125,315,150,'Runtime guard',['Qcur-0 and N = 1','Live W X Y ranges'])
    box(1070,15,325,150,'Stock Metal',['Default and fallback','All other nodes'])
    box(1070,245,325,150,'Opt in Q4',['Four row helper','One SIMD group'],'#E6EFF8')
    box(1450,125,300,150,'GPU completion',['Residual and logits','Framework continues'])
    arrow([(285,200),(335,200)]);arrow([(645,200),(700,200)])
    arrow([(1015,175),(1040,175),(1040,90),(1070,90)])
    arrow([(1015,225),(1040,225),(1040,320),(1070,320)])
    arrow([(1395,90),(1420,90),(1420,175),(1450,175)])
    arrow([(1395,320),(1420,320),(1420,225),(1450,225)])
    draw.text((width/2,420),'One eligible projection is replaced; the surrounding Transformer operations remain stock.',font=regular,fill='#333333',anchor='mt')
    svg.append('<text x="880" y="447" text-anchor="middle" font-family="Arial" font-size="28">One eligible projection is replaced; the surrounding Transformer operations remain stock.</text></svg>')
    path=HERE/'assets/execution-flow.png';path.parent.mkdir(exist_ok=True)
    im.save(path,dpi=(300,300));path.with_suffix('.svg').write_text('\n'.join(svg))


def parse(text):
    lines=text.splitlines();out=[];i=0
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line=='<!-- page -->':out.append(('break',None));i+=1;continue
        if line.startswith('```'):
            i+=1;block=[]
            while i<len(lines) and not lines[i].startswith('```'):block.append(lines[i]);i+=1
            out.append(('code','\n'.join(block)));i+=1;continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                row=[x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch('[: -]+',x) for x in row):rows.append(row)
                i+=1
            out.append(('table',rows));continue
        m=re.match(r'!\[(.*?)\]\((.*?)\)',line)
        if m:out.append(('image',(m[1],m[2])));i+=1;continue
        m=re.match(r'^(#{1,3}) (.*)',line)
        if m:out.append(('h'+str(len(m[1])),m[2]));i+=1;continue
        if line.startswith('- '):out.append(('bullet',line[2:]));i+=1;continue
        para=[line];i+=1
        while i<len(lines) and lines[i].strip() and not re.match(r'^(#|\||```|<!--|!\[|- )',lines[i].strip()):para.append(lines[i].strip());i+=1
        out.append(('p',' '.join(para)))
    return out


def inline_xml(text):
    text=escape(text)
    text=re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)',r'<link href="\2" color="#204B70">\1</link>',text)
    text=re.sub(r'`([^`]+)`',r'<font name="Mono" size="8.4">\1</font>',text)
    return re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',text)


def widths(rows, width):
    n=len(rows[0]);header=' '.join(rows[0])
    if n==2:
        fractions=[.44,.56] if 'file' in header.lower() or 'Source location' in header else [.32,.68]
    elif n==3:
        if rows[0][0]=='Route' and 'Implementation' in header:fractions=[.08,.46,.46]
        elif rows[0][0]=='Day':fractions=[.06,.29,.65]
        elif 'Risk' in header:fractions=[.26,.26,.48]
        elif 'Proposed acceptance' in header:fractions=[.20,.48,.32]
        else:fractions=[.25,.35,.40]
    elif n==4:
        fractions=[.10,.33,.39,.18] if rows[0][0]=='Route' and 'Full FFN GPU' in header else [.24,.25,.25,.26]
    elif n==5:
        if 'p50' in header:fractions=[.20,.23,.23,.20,.14]
        elif 'Grid' in header:fractions=[.22,.15,.18,.12,.33]
        elif 'Metal placement' in header:fractions=[.27,.08,.20,.25,.20]
        else:fractions=[.18,.24,.18,.18,.22]
    else:fractions=[1/n]*n
    return [width*x for x in fractions]


def numeric_cell(text):
    return bool(re.fullmatch(r'[-+0-9., /%e]+|Pass',text))


def make_pdf(blocks,path):
    for name,key in [('Body','regular'),('BodyBold','bold'),('BodyItalic','italic'),('Mono','mono')]:
        font=font_path(key)
        if font:pdfmetrics.registerFont(TTFont(name,str(font)))
    pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='BodyItalic',boldItalic='BodyBold')
    body=ParagraphStyle('Body',fontName='Body',fontSize=10.1,leading=13.6,spaceAfter=7,textColor=colors.black)
    styles={'p':body,'bullet':ParagraphStyle('Bullet',parent=body,leftIndent=9,firstLineIndent=-6,spaceAfter=6),
        'h1':ParagraphStyle('Title',parent=body,fontName='BodyBold',fontSize=25,leading=29,spaceAfter=12,keepWithNext=True),
        'h2':ParagraphStyle('Heading1',parent=body,fontName='BodyBold',fontSize=17,leading=21,spaceBefore=4,spaceAfter=12,keepWithNext=True),
        'h3':ParagraphStyle('Heading2',parent=body,fontName='BodyBold',fontSize=11.5,leading=15,spaceBefore=8,spaceAfter=7,keepWithNext=True)}
    cell=ParagraphStyle('Cell',parent=body,fontSize=8.9,leading=11.6,spaceAfter=0)
    head=ParagraphStyle('Head',parent=cell,fontName='BodyBold')
    code=ParagraphStyle('Code',parent=body,fontName='Mono',fontSize=8.5,leading=11.5,leftIndent=4,spaceBefore=4,spaceAfter=9)
    doc=SimpleDocTemplate(str(path),pagesize=A4,rightMargin=43,leftMargin=43,topMargin=44,bottomMargin=45,title='Mobile BitNet Inference Technical Proposal',author='Candidate',pageCompression=1)
    story=[]
    for kind,value in blocks:
        if kind=='break':story.append(PageBreak())
        elif kind=='table':
            data=[[Paragraph(inline_xml(x), head if j==0 else ParagraphStyle('Numeric',parent=cell,alignment=TA_CENTER) if numeric_cell(x) else cell) for x in row] for j,row in enumerate(value)]
            table=Table(data,colWidths=widths(value,doc.width),repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E7EEF5')),('GRID',(0,0),(-1,-1),.5,colors.HexColor('#D9D9D9')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
            story.extend([table,Spacer(1,9)])
        elif kind=='code':story.append(XPreformatted(escape(value),code))
        elif kind=='image':
            source=HERE/value[1];im=Image.open(source);w=doc.width;h=w*im.height/im.width
            story.extend([PDFImage(str(source),width=w,height=h),Spacer(1,8)])
        else:story.append(Paragraph(('&#8226; ' if kind=='bullet' else '')+inline_xml(value),styles[kind]))
    def page(canvas,document):
        canvas.saveState();canvas.setFont('Body',8);canvas.setFillColor(colors.black)
        canvas.drawString(43,A4[1]-25,'Mobile BitNet inference technical proposal')
        canvas.drawRightString(A4[0]-43,25,f'Draft for author review     {document.page}')
        canvas.restoreState()
    doc.build(story,onFirstPage=page,onLaterPages=page)


def add_inline(p,text):
    pattern=re.compile(r'(\[[^\]]+\]\(https?://[^)]+\)|`[^`]+`|\*\*[^*]+\*\*)')
    for part in pattern.split(text):
        if not part:continue
        m=re.fullmatch(r'\[([^\]]+)\]\((https?://[^)]+)\)',part)
        if m:
            h=OxmlElement('w:hyperlink');h.set(qn('r:id'),p.part.relate_to(m[2],'http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink',is_external=True))
            r=OxmlElement('w:r');props=OxmlElement('w:rPr');color=OxmlElement('w:color');color.set(qn('w:val'),'204B70');props.append(color);r.append(props);t=OxmlElement('w:t');t.text=m[1];r.append(t);h.append(r);p._p.append(h)
        else:
            code=part.startswith('`');bold=part.startswith('**')
            r=p.add_run(part[1:-1] if code else part[2:-2] if bold else part)
            if code:r.font.name='Courier New';r.font.size=Pt(8.4)
            if bold:r.bold=True


def make_docx(blocks,path):
    doc=Document()
    for border in doc.styles.element.xpath('.//w:pBdr'):
        border.getparent().remove(border)
    sec=doc.sections[0];sec.page_width=Inches(A4[0]/72);sec.page_height=Inches(A4[1]/72)
    sec.left_margin=sec.right_margin=Inches(43/72);sec.top_margin=Inches(44/72);sec.bottom_margin=Inches(45/72)
    sec.header_distance=sec.footer_distance=Inches(.25)
    for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Heading 3']:
        st=doc.styles[name];st.font.name='Arial';st.font.color.rgb=RGBColor(0,0,0)
        st.element.get_or_add_rPr().append(OxmlElement('w:rFonts'))
        st.element.rPr.rFonts.set(qn('w:ascii'),'Arial');st.element.rPr.rFonts.set(qn('w:hAnsi'),'Arial')
    normal=doc.styles['Normal'];normal.font.size=Pt(10.1);normal.paragraph_format.line_spacing=1.12;normal.paragraph_format.space_after=Pt(7)
    for name,size in [('Title',25),('Heading 1',17),('Heading 2',11.5)]:
        doc.styles[name].font.size=Pt(size);doc.styles[name].font.bold=True
        doc.styles[name].paragraph_format.space_after=Pt(10);doc.styles[name].paragraph_format.keep_with_next=True
    header=sec.header.paragraphs[0];header.text='Mobile BitNet inference technical proposal';header.runs[0].font.size=Pt(8)
    footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
    run=footer.add_run('Draft for author review     ');run.font.size=Pt(8)
    field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
    page_before = False
    for kind,value in blocks:
        if kind=='break':page_before = True
        elif kind=='table':
            table=doc.add_table(rows=0,cols=len(value[0]));table.alignment=WD_TABLE_ALIGNMENT.CENTER;table.autofit=False
            ws=widths(value,(A4[0]-86)/72)
            for col,w in zip(table.columns,ws):col.width=Inches(w)
            for grid,w in zip(table._tbl.tblGrid.gridCol_lst,ws):grid.set(qn('w:w'),str(round(w*1440)))
            for j,row in enumerate(value):
                cells=table.add_row().cells
                trpr=table.rows[-1]._tr.get_or_add_trPr();trpr.append(OxmlElement('w:cantSplit'))
                if j==0:trpr.append(OxmlElement('w:tblHeader'))
                for k,(cell,text) in enumerate(zip(cells,row)):
                    cell.width=Inches(ws[k]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                    pr=cell._tc.get_or_add_tcPr();borders=OxmlElement('w:tcBorders')
                    for side in ['top','left','bottom','right']:
                        e=OxmlElement('w:'+side);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');borders.append(e)
                    pr.append(borders);margins=OxmlElement('w:tcMar')
                    for side,v in [('top',75),('bottom',75),('left',80),('right',80)]:
                        e=OxmlElement('w:'+side);e.set(qn('w:w'),str(v));e.set(qn('w:type'),'dxa');margins.append(e)
                    pr.append(margins)
                    if j==0:
                        sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'E7EEF5');pr.append(sh)
                    p=cell.paragraphs[0];p.paragraph_format.space_after=Pt(0);p.paragraph_format.line_spacing=1.05
                    add_inline(p,text)
                    if j>0 and numeric_cell(text):p.alignment=WD_ALIGN_PARAGRAPH.CENTER
                    for r in p.runs:r.font.size=Pt(8.7 if r.font.name!='Courier New' else 8.0);r.bold=(j==0) or bool(r.bold)
            doc.add_paragraph().paragraph_format.space_after=Pt(1)
        elif kind=='code':
            for line in value.splitlines():
                p=doc.add_paragraph();p.paragraph_format.space_after=Pt(0);p.paragraph_format.line_spacing=1.0
                r=p.add_run(line);r.font.name='Courier New';r.font.size=Pt(8.3)
            doc.add_paragraph().paragraph_format.space_after=Pt(1)
        elif kind=='image':
            p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
            r=p.add_run();r.add_picture(str(HERE/value[1]),width=Inches((A4[0]-86)/72))
            r._r.xpath('.//wp:docPr')[0].set('descr',value[0])
        else:
            p=doc.add_paragraph(style={'h1':'Title','h2':'Heading 1','h3':'Heading 2'}.get(kind,'Normal'))
            if page_before:
                p.paragraph_format.page_break_before = True
                page_before = False
            add_inline(p,('- ' if kind=='bullet' else '')+value)
    doc.core_properties.title='Mobile BitNet Inference Technical Proposal';doc.core_properties.author='Candidate';doc.core_properties.subject='Metal proof of concept and Vulkan portability plan'
    doc.save(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);figure();blocks=parse((HERE/'PROPOSAL.md').read_text())
    make_pdf(blocks,args.output/'BitNet_Mobile_GPU_Proposal.pdf');make_docx(blocks,args.output/'BitNet_Mobile_GPU_Proposal.docx')
    print('Created PDF and DOCX from one editable source; inference not executed.')

if __name__=='__main__':main()
