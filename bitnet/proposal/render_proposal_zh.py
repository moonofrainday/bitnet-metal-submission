#!/usr/bin/env python3
"""Render the faithful Chinese companion without modifying English artifacts."""
import argparse
from html import escape
from pathlib import Path
import re

from PIL import Image
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, PageBreak, Spacer, XPreformatted, Image as PDFImage

import render_proposal as en

HERE=Path(__file__).resolve().parent
TITLE='移动 GPU BitNet 推理技术方案'
SUBTITLE='Metal 概念验证与 Vulkan 跨平台适配计划'
FOOTER='技术方案 中文版'


def set_fonts(rpr,family):
    # Explicit names avoid duplicate/theme font mappings in headless renderers.
    for fonts in list(rpr.findall(qn('w:rFonts'))):rpr.remove(fonts)
    fonts=OxmlElement('w:rFonts')
    for field in ('ascii','hAnsi','eastAsia','cs'):fonts.set(qn('w:'+field),family)
    fonts.set(qn('w:hint'),'eastAsia');rpr.insert(0,fonts)
    lang=rpr.find(qn('w:lang'))
    if lang is None:lang=OxmlElement('w:lang');rpr.append(lang)
    lang.set(qn('w:val'),'zh-CN');lang.set(qn('w:eastAsia'),'zh-CN')


def font_run(run,east_asia='Arial Unicode MS'):
    run.font.color.rgb=RGBColor(0,0,0)
    family='Courier New' if run.font.name=='Courier New' else east_asia
    set_fonts(run._r.get_or_add_rPr(),family)


def make_docx(blocks,path):
    en.make_docx(blocks,path)
    doc=Document(path)
    for st in doc.styles:
        if not hasattr(st,'font'):continue
        family='Heiti SC' if st.name in ('Title','Heading 1','Heading 2','Heading 3') else 'Arial Unicode MS'
        set_fonts(st.element.get_or_add_rPr(),family)
        st.font.color.rgb=RGBColor(0,0,0)
    doc.styles['Normal'].font.size=Pt(10.8)
    doc.styles['Normal'].paragraph_format.line_spacing=Pt(15.2)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.line_spacing=Pt(12.7)
                    for run in paragraph.runs:
                        font_run(run);run.font.size=Pt(8.0 if run.font.name=='Courier New' else 9.1)
    for paragraph in doc.paragraphs:
        if paragraph._p.xpath('.//w:drawing'):
            paragraph.paragraph_format.line_spacing=1.0
        elif paragraph.style.name=='Title':paragraph.paragraph_format.line_spacing=Pt(31)
        elif paragraph.style.name=='Heading 1':paragraph.paragraph_format.line_spacing=Pt(23)
        elif paragraph.style.name=='Heading 2':paragraph.paragraph_format.line_spacing=Pt(16)
        elif not paragraph.text and not paragraph._p.xpath('.//w:br'):
            paragraph.paragraph_format.line_spacing=Pt(5);paragraph.paragraph_format.space_after=Pt(3)
        elif paragraph.runs and all(r.font.name=='Courier New' for r in paragraph.runs):
            paragraph.paragraph_format.line_spacing=Pt(11.5)
        for run in paragraph.runs:
            font_run(run,'Heiti SC' if paragraph.style.name in ('Title','Heading 1','Heading 2','Heading 3') else 'Arial Unicode MS')
    for section in doc.sections:
        header=section.header.paragraphs[0];header.clear();r=header.add_run(TITLE);r.font.size=Pt(8);font_run(r)
        footer=section.footer.paragraphs[0];footer.clear();footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
        r=footer.add_run(FOOTER+'     ');r.font.size=Pt(8);font_run(r)
        field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
    doc.core_properties.title=TITLE;doc.core_properties.subject=SUBTITLE;doc.core_properties.author='Candidate'
    doc.save(path)


def make_pdf(blocks,path,args):
    pdfmetrics.registerFont(TTFont('ZHBody',str(args.cjk_font),subfontIndex=args.cjk_index))
    pdfmetrics.registerFont(TTFont('ZHBold',str(args.cjk_bold_font),subfontIndex=args.cjk_bold_index))
    pdfmetrics.registerFont(TTFont('Mono',str(en.font_path('mono'))))
    pdfmetrics.registerFontFamily('ZHBody',normal='ZHBody',bold='ZHBold',italic='ZHBody',boldItalic='ZHBold')
    required=set(''.join(str(v) for _,v in blocks))
    missing=[c for c in required if '\u3400'<=c<='\u9fff' and ord(c) not in pdfmetrics.getFont('ZHBody').face.charToGlyph]
    if missing:raise ValueError('CJK font lacks characters: '+''.join(sorted(missing)))
    body=ParagraphStyle('ZHBody',fontName='ZHBody',fontSize=10.8,leading=14.8,spaceAfter=6,textColor=colors.black,wordWrap='CJK')
    styles={'p':body,'bullet':ParagraphStyle('Bullet',parent=body,leftIndent=8,firstLineIndent=-5,spaceAfter=5),
        'h1':ParagraphStyle('Title',parent=body,fontName='ZHBold',fontSize=25,leading=31,spaceAfter=12,keepWithNext=True),
        'h2':ParagraphStyle('Heading1',parent=body,fontName='ZHBold',fontSize=17,leading=23,spaceBefore=4,spaceAfter=12,keepWithNext=True),
        'h3':ParagraphStyle('Heading2',parent=body,fontName='ZHBold',fontSize=11.5,leading=16,spaceBefore=8,spaceAfter=7,keepWithNext=True)}
    cell=ParagraphStyle('Cell',parent=body,fontSize=9.1,leading=12.4,spaceAfter=0)
    head=ParagraphStyle('Head',parent=cell,fontName='ZHBold')
    center=ParagraphStyle('Numeric',parent=cell,alignment=TA_CENTER)
    code=ParagraphStyle('Code',parent=body,fontName='Mono',fontSize=8.5,leading=11.5,leftIndent=4,spaceBefore=4,spaceAfter=9)
    doc=SimpleDocTemplate(str(path),pagesize=A4,rightMargin=43,leftMargin=43,topMargin=44,bottomMargin=45,title=TITLE,author='Candidate',pageCompression=1)
    story=[]
    for kind,value in blocks:
        if kind=='break':story.append(PageBreak())
        elif kind=='table':
            data=[[Paragraph(en.inline_xml(x),head if j==0 else center if en.numeric_cell(x) or x=='通过' else cell) for x in row] for j,row in enumerate(value)]
            table=Table(data,colWidths=en.widths(value,doc.width),repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E7EEF5')),('GRID',(0,0),(-1,-1),.5,colors.HexColor('#D9D9D9')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
            story.extend([table,Spacer(1,9)])
        elif kind=='code':story.append(XPreformatted(escape(value),code))
        elif kind=='image':
            source=HERE/value[1];im=Image.open(source);width=doc.width
            story.extend([PDFImage(str(source),width=width,height=width*im.height/im.width),Spacer(1,8)])
        else:story.append(Paragraph(('&#8226; ' if kind=='bullet' else '')+en.inline_xml(value),styles[kind]))
    def page(canvas,document):
        canvas.saveState();canvas.setFont('ZHBody',8);canvas.setFillColor(colors.black)
        canvas.drawString(43,A4[1]-25,TITLE)
        canvas.drawRightString(A4[0]-43,25,f'{FOOTER}     {document.page}');canvas.restoreState()
    doc.build(story,onFirstPage=page,onLaterPages=page)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--docx-only',action='store_true',help='Render only the editable Word companion')
    parser.add_argument('--cjk-font',type=Path,default=Path('/System/Library/Fonts/Supplemental/Arial Unicode.ttf'))
    parser.add_argument('--cjk-index',type=int,default=0)
    parser.add_argument('--cjk-bold-font',type=Path,default=Path('/System/Library/Fonts/STHeiti Medium.ttc'))
    parser.add_argument('--cjk-bold-index',type=int,default=1)
    args=parser.parse_args()
    if not args.cjk_font.is_file() or not args.cjk_bold_font.is_file():
        parser.error('Supply installed CJK TTF/TTC fonts with --cjk-font and --cjk-bold-font; fonts are not bundled.')
    blocks=en.parse((HERE/'PROPOSAL_ZH.md').read_text());english=en.parse((HERE/'PROPOSAL.md').read_text())
    zh_tables=[v for k,v in blocks if k=='table'];en_tables=[v for k,v in english if k=='table']
    if len(zh_tables)!=len(en_tables):raise ValueError('Table count differs from English source')
    width_map={tuple(z[0]):e for z,e in zip(zh_tables,en_tables)}
    original_widths=en.widths
    en.widths=lambda rows,width:original_widths(width_map.get(tuple(rows[0]),rows),width)
    args.output.mkdir(parents=True,exist_ok=True)
    if not args.docx_only:make_pdf(blocks,args.output/'BitNet_Mobile_GPU_Proposal_ZH.pdf',args)
    make_docx(blocks,args.output/'BitNet_Mobile_GPU_Proposal_ZH.docx')
    print('Chinese PDF and DOCX created; English sources/artifacts and figure bytes unchanged.')

if __name__=='__main__':main()
