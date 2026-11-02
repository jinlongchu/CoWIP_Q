"""Build a MADQN-based architecture overview for the M4 booklet."""
from pathlib import Path
import math, json, hashlib
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader, PdfWriter
from pypdf.generic import NameObject, TextStringObject
import pdfplumber

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'outputs'/'M4_Algorithm_Booklets'
SOURCE=OUT/'M4_No_PI_Architecture_and_Improvements.pdf'
RESULT=OUT/'M4_No_PI_MADQN_Based_Overview.pdf'
WORK=ROOT/'work'/'m4_madqn_overview_qa'
WORK.mkdir(parents=True,exist_ok=True)
FIRST=WORK/'first_page.pdf'
pdfmetrics.registerFont(TTFont('CN',str(ROOT/'work/fonts/DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('CNB',str(ROOT/'work/fonts/DejaVuSans.ttf')))
pdfmetrics.registerFontFamily('CN',normal='CN',bold='CNB',italic='CN',boldItalic='CNB')
W,H=landscape(A4)
NAVY='#183453';INK='#263644';MUTED='#627383';GRAY='#66717C'
TEAL='#087C80';BLUE='#326BB1';ORANGE='#B96F25';PURPLE='#7555A2';LINE='#CBD5DE'
WHITE='#FFFFFF';PALE='#F6F8FA'
C=canvas.Canvas(str(FIRST),pagesize=(W,H),pageCompression=1)
C.setTitle('New M4 Algorithm Overview - Based on Original MADQN')
boxes=[];paragraphs=[]

def color(s): return colors.HexColor(s)

def math_markup(s):
    for char,tag in {'⁺':'<super>+</super>','⁻':'<super>-</super>',
                     '₁':'<sub>1</sub>','₂':'<sub>2</sub>','₃':'<sub>3</sub>'}.items():
        s=s.replace(char,tag)
    return s

def txt(x,y,s,size=9,color_=INK,bold=False):
    face='CNB' if bold else 'CN'
    assert x+pdfmetrics.stringWidth(s,face,size)<W-5,(s,'off-page')
    C.setFillColor(color(color_));C.setFont(face,size);C.drawString(x,H-y-size*.86,s)

def paragraph(x,y,w,s,size=9,lead=None,color_=INK,bold=False,align=0,maxh=None):
    st=ParagraphStyle('p',fontName='CNB' if bold else 'CN',fontSize=size,leading=lead or size*1.35,
                      textColor=color(color_),wordWrap='LTR',alignment=align)
    p=Paragraph(math_markup(s),st);_,ph=p.wrap(w,700)
    assert maxh is None or ph<=maxh+.1,(s,ph,maxh)
    p.drawOn(C,x,H-y-ph);paragraphs.append((x,y,w,ph,s))
    return ph

def rect(x,y,w,h,stroke=GRAY,fill=WHITE,dash=False,r=5,lw=.8):
    C.saveState();C.setStrokeColor(color(stroke));C.setFillColor(color(fill));C.setLineWidth(lw)
    if dash:C.setDash(3,3)
    C.roundRect(x,H-y-h,w,h,r,stroke=1,fill=1);C.restoreState()

def node(x,y,w,h,body,stroke=GRAY,size=9,fill=WHITE,dash=False,bold=False):
    rect(x,y,w,h,stroke,fill,dash)
    while True:
        st=ParagraphStyle('node',fontName='CNB' if bold else 'CN',fontSize=size,leading=size*1.22,
                          textColor=color(stroke if stroke!=GRAY else INK),wordWrap='LTR',alignment=1)
        p=Paragraph(math_markup(body),st);_,ph=p.wrap(w-10,600)
        if ph<=h-6:break
        size-=.2
        assert size>=7.4,(body,ph,h)
    p.drawOn(C,x+6,H-y-(h+ph)/2)
    boxes.append((x,y,w,h,body))

def circle(x,y,r,body,stroke=GRAY,size=9):
    C.setStrokeColor(color(stroke));C.setFillColor(color(WHITE));C.setLineWidth(.8)
    C.circle(x,H-y,r,stroke=1,fill=1)
    paragraph(x-r,y-size*.7,2*r,body,size,align=1)

def arrow(pts,stroke=GRAY,dash=False,lw=.85,head=4):
    C.saveState();C.setStrokeColor(color(stroke));C.setFillColor(color(stroke));C.setLineWidth(lw)
    if dash:C.setDash(2.5,2.5)
    p=C.beginPath();p.moveTo(pts[0][0],H-pts[0][1])
    for x,y in pts[1:]:p.lineTo(x,H-y)
    C.drawPath(p);C.setDash()
    x,y=pts[-1];a=math.atan2(y-pts[-2][1],x-pts[-2][0])
    p=C.beginPath();p.moveTo(x,H-y)
    for d in (-.5,.5):p.lineTo(x-head*math.cos(a+d),H-(y-head*math.sin(a+d)))
    p.close();C.drawPath(p,stroke=0,fill=1);C.restoreState()

txt(32,17,'M4 / PI OFF',9,TEAL,True)
txt(662,18,'Based on original MADQN',8.8,MUTED)
txt(32,36,'New M4: Improvements to the MADQN Architecture',20.5,NAVY,True)
txt(33,66,'Retain joint interaction and Double DQN; add global features, GRU, QMIX, 3-step, PER and five-model independent Q ensembling.',9.1,MUTED)
txt(33,82,'Legend: black/gray = retained MADQN; colors 1-6 = added/modified; small dashed boxes = targets or target computation.',8.3,MUTED)

# A: same three-branch / joint-action / environment / feedback / replay skeleton.
rect(24,99,794,194,NAVY,dash=True,r=10)
txt(36,107,'A  Online interaction: joint actions and shared team reward',10.5,NAVY,True)
txt(514,108,'Member k shown; all five members train independently',8.5,MUTED)

# Observation feedback loop above the agent lanes.
arrow([(677,184),(677,130),(29,130),(29,253)],GRAY,head=0)
txt(337,119,'Feedback O(t+1) updates observations and history windows',8.1,MUTED)
paragraph(80,134,145,'① Append o<sub>i</sub>(45) + g(24)',8.1,color_=TEAL,bold=True)
ys=[163,208,253]
for i,y in enumerate(ys,1):
    circle(50,y,15,f'o<sub>{i}</sub>(t)',size=8.8)
    arrow([(29,y),(35,y)])
    node(84,y-16,99,32,f'History H<sub>{i}</sub>(t)<br/>4 × 69 dimensions',TEAL,8.6,fill='#F3FAF9')
    arrow([(65,y),(84,y)])
    arrow([(183,y),(203,y)],TEAL)
node(203,147,62,122,'② Shared<br/><b>GRU</b><br/>69 → 64<br/><br/>Encode each<br/>unit history',TEAL,9,fill='#F3FAF9')
for i,y in enumerate(ys,1):
    node(285,y-16,104,32,f"""Unit {i} Q head<br/>Q<sub>{i}</sub>(H<sub>{i}</sub>, a)""",TEAL,8.7)
    arrow([(265,y),(285,y)],TEAL)
    circle(418,y,14,f'a<sub>{i}</sub>(t)',size=8.7)
    arrow([(389,y),(404,y)])
node(457,185,72,49,'Joint action<br/><b>A(t)</b><br/>[a₁, a₂, a₃]',GRAY,9)
arrow([(432,163),(443,163),(443,197),(457,197)])
arrow([(432,208),(457,208)])
arrow([(432,253),(443,253),(443,222),(457,222)])
node(550,185,72,49,'Shared<br/><b>Environment</b>',GRAY,8.8)
arrow([(529,209.5),(550,209.5)])
node(644,184,67,51,'One-step feedback<br/>r, O(t+1), d',GRAY,8.6)
arrow([(622,209.5),(644,209.5)])
node(733,143,72,47,'④ <b>3-step</b><br/>Accumulate R(n)<br/>Usually n = 3',ORANGE,8.7,fill='#FEF8F0')
arrow([(711,209.5),(721,209.5),(721,166.5),(733,166.5)])
node(733,216,72,54,'⑤ <b>PER buffer Dk</b><br/>Joint experience<br/>Shared by 3 units',ORANGE,8.5,fill='#FEF8F0')
arrow([(769,190),(769,216)],ORANGE)
txt(84,275,'g: 24-D summary of the three unit observations',8,TEAL)
txt(346,275,'Legal mask + ε-greedy; separate Q-head parameters; no PI',8.1,MUTED)

# Replay samples feed both current-value and successor-target branches.
arrow([(769,270),(769,309),(18,309),(18,417),(37,417)],GRAY)
txt(203,295,'⑤ PER draws shared joint samples (all three units) and importance weights w',8.1,ORANGE)

# B: original Double DQN branching, upgraded to a joint QMIX loss.
rect(24,313,794,188,NAVY,dash=True,r=10)
txt(36,321,'B  Replay and stable TD targets: individual losses → QMIX team loss',10.4,NAVY,True)

node(37,362,127,114,'One joint replay sample<br/><br/>H, g, A, R(n),<br/>H⁺, g⁺, d, γⁿ<br/><br/>H⁺ = H(t+n)',GRAY,8.8,fill=PALE)
node(187,362,139,39,'Online: current-action values<br/>q<sub>i</sub> = Q<sub>i</sub>(H<sub>i</sub>, a<sub>i</sub>), i = 1,2,3',GRAY,8.7)
arrow([(164,381.5),(187,381.5)])
node(351,362,142,39,'③ Online <b>QMIX</b> + g<br/>[q₁, q₂, q₃] → Q<sub>tot</sub>',BLUE,9.1,fill='#F3F7FC')
arrow([(326,381.5),(351,381.5)])
node(677,353,128,54,'Team TD error / loss<br/>δ = Q<sub>tot</sub> - y<br/>⑤ L = mean[w × Huber(δ)]',ORANGE,8.7,fill='#FEF8F0')
arrow([(493,381.5),(677,381.5)])
txt(529,368,'Current team prediction',8.4,MUTED)

# Gradient arrows, online parameters only.
arrow([(741,353),(741,347),(256,347),(256,362)],ORANGE,dash=True)
arrow([(423,347),(423,362)],ORANGE,dash=True)
txt(239,335,'Backpropagate only through online GRU, 3 Q heads and QMIX',8.2,ORANGE)

node(187,431,128,45,'Online: select successor actions<br/>a*<sub>i</sub> = argmax<sub>legal a</sub> Q<sub>i</sub>(H⁺<sub>i</sub>, a)',GRAY,8.5)
arrow([(164,453.5),(187,453.5)])
node(337,431,132,45,'Target Q: evaluate<br/>q⁻<sub>i</sub> = Q⁻<sub>i</sub>(H⁺<sub>i</sub>, a*<sub>i</sub>)',GRAY,8.8,dash=True)
arrow([(315,453.5),(337,453.5)])
node(491,431,116,45,'③ Target <b>QMIX</b> + g⁺<br/>[q⁻₁, q⁻₂, q⁻₃] → Q⁻<sub>tot</sub>',BLUE,8.4,dash=True)
arrow([(469,453.5),(491,453.5)])
node(630,425,175,55,'④ Multi-step TD target y<br/>R(n) + γⁿ(1-d) Q⁻<sub>tot</sub><br/>Stop gradient; γ = 0.99',ORANGE,8.8,dash=True)
arrow([(607,453.5),(630,453.5)])
arrow([(741,425),(741,407)],ORANGE)

# Both types of target parameters are copied from their corresponding online networks.
arrow([(256,401),(256,419),(404,419),(404,431)],BLUE,dash=True)
arrow([(423,401),(423,419),(549,419),(549,431)],BLUE,dash=True)
txt(293,406,'Periodically hard-copy online parameters to targets',7.9,BLUE)
txt(39,485,'R(n), γⁿ and d come from the same replay sample. Targets sync every 250 gradient updates. Successor a* is used only in targets, not sent to the environment.',8.2,MUTED)

# TD error is returned to the same member's PER buffer, not shared across models.
arrow([(805,373),(825,373),(825,243),(805,243)],ORANGE)
C.saveState();C.translate(835,H-366);C.rotate(90)
C.setFont('CN',8);C.setFillColor(color(ORANGE));C.drawString(0,0,'|δ| updates PER priorities');C.restoreState()

# C: deployment ensemble is separate from the single-member training graph.
rect(24,510,794,43,PURPLE,fill='#FAF8FD',dash=True,r=7)
txt(36,521,'⑥ Five-model independent Q ensemble',10,PURPLE,True)
node(243,517,258,24,'Same H → 5 independent members → average Q per unit/action',PURPLE,8.5)
node(523,517,282,24,'Legal argmax → joint action → environment (no exploration or PI)',PURPLE,8.5)
arrow([(501,529),(523,529)],PURPLE)
txt(245,543,'No QMIX, targets or replay at execution; g is still broadcast. The five training replay buffers are independent.',7.7,MUTED)

C.setStrokeColor(color(LINE));C.setLineWidth(.7);C.line(32,32,W-32,32)
txt(32,H-24,'CONWIP intelligent control · MADQN-based overview · 2026.10',8,MUTED)
C.setFont('CN',8);C.setFillColor(color(MUTED));C.drawRightString(W-32,15,'01 / 20')
C.save()

# Keep the original page object / bookmark destinations; replace its graphics and fonts only.
source_bytes=SOURCE.read_bytes()
old=PdfReader(SOURCE);fresh=PdfReader(FIRST)
writer=PdfWriter();writer.clone_document_from_reader(old)
page=writer.pages[0]
for key in ('/Contents','/Resources','/MediaBox','/CropBox','/Rotate'):
    if key in fresh.pages[0]:page[NameObject(key)]=fresh.pages[0].raw_get(key).clone(writer)
    elif key in page and key in ('/CropBox','/Rotate'):del page[key]
writer.add_metadata({'/Title':'New M4 Independent Q Ensemble (PI Off) - MADQN-Based Overview',
                     '/Subject':'MADQN-based architecture overview, module explanations and fixed-seed ablation results'})
outline_root=writer.root_object.get('/Outlines')
if outline_root:
    first_outline=outline_root.get_object().get('/First')
    if first_outline:first_outline.get_object()[NameObject('/Title')]=TextStringObject('01  New M4: Improvements to the MADQN Architecture')
with RESULT.open('wb') as f:writer.write(f)
assert SOURCE.read_bytes()==source_bytes,'Original booklet was unexpectedly modified'
new=PdfReader(RESULT)
assert len(new.pages)==len(old.pages)==20
for i in range(1,20):
    assert new.pages[i].get_contents().get_data()==old.pages[i].get_contents().get_data(),i
    assert new.pages[i].extract_text()==old.pages[i].extract_text(),i
assert len(new.outline)==len(old.outline)==20
assert len(new.pages[-1].get('/Annots',[]))==4
with pdfplumber.open(RESULT) as doc:
    for i,p in enumerate(doc.pages):
        assert not any(c['text'] in ('\x00','\ufffd') for c in p.chars),(i,'missing glyph')
        assert not any(c['x0']<0 or c['x1']>p.width or c['top']<0 or c['bottom']>p.height for c in p.chars),(i,'clipping')
    first_text=doc.pages[0].extract_text()
    for term in ['MADQN','GRU','QMIX','3-step','PER','Five-model','no PI']:assert term.lower() in first_text.lower(),term
report={'output':str(RESULT),'pages':20,'changed_pages':[1],'pages_2_to_20':'content bytes and text unchanged',
        'original_preserved':True,'outline_entries':20,'reference_links':4,'glyph_and_boundary_checks':'pass',
        'sha256':hashlib.sha256(RESULT.read_bytes()).hexdigest()}
(WORK/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=True))
