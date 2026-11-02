from __future__ import annotations

import json, math
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'outputs' / 'm4_no_pi_ablation'
OUT = ROOT / 'outputs' / 'M4_Algorithm_Booklets'
OUT.mkdir(parents=True, exist_ok=True)
PDF = OUT / 'M4_No_PI_Architecture_and_Improvements.pdf'
pdfmetrics.registerFont(TTFont('CN',str(ROOT/'work/fonts/DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('CNB',str(ROOT/'work/fonts/DejaVuSans.ttf')))
pdfmetrics.registerFontFamily('CN', normal='CN', bold='CNB', italic='CN', boldItalic='CNB')

W, H = landscape(A4)
NAVY = '#183453'; INK = '#253A4D'; MUTED = '#607486'; TEAL = '#147D80'
BLUE = '#3B70B2'; PURPLE = '#7760A2'; ORANGE = '#C47730'; RED = '#B75250'
PALE = '#F3F6F8'; LINE = '#CDD7DF'; WHITE = '#FFFFFF'
C = canvas.Canvas(str(PDF), pagesize=(W,H), pageCompression=1)
C.setTitle('M4 Independent Q Ensemble: Architecture and Improvements with PI Off')
C.setAuthor('')
C.setCreator('')
C.setSubject('Global features, GRU, QMIX, 3-step, PER and five-model independent Q ensembling')
LAYOUT = []; PAGE = 0

summaries=json.loads((DATA/'comparison_summary.json').read_text(encoding='utf-8'))
summary={(r['config'],r['method']):r for r in summaries if r['split']=='test'}
contrasts=json.loads((DATA/'paired_contrasts.json').read_text(encoding='utf-8'))
contrast={(r['ablation'],r['method'],r['metric']):r for r in contrasts}
base=summary['no_pi','q_ensemble_5']
manifest=json.loads((DATA/'training_manifest.json').read_text(encoding='utf-8'))
assert len(manifest)==30 and len(summary)==12

def col(h): return colors.HexColor(h)

def text(x,y,s,size=11,color=INK,bold=False,max_width=None):
    face='CNB' if bold else 'CN'
    limit=max_width or (365 if PAGE>1 and x<100 and y>300 and size>9 else W-x-32)
    while pdfmetrics.stringWidth(str(s),face,size)>limit and size>8:
        size-=.2
    C.setFillColor(col(color)); C.setFont('CNB' if bold else 'CN',size)
    C.drawString(x,H-y-size*.85,str(s))

def para(x,y,w,s,size=11,leading=None,color=INK,bold=False,align=0,maxh=None):
    st=ParagraphStyle('p',fontName='CNB' if bold else 'CN',fontSize=size,
        leading=leading or size*1.5,textColor=col(color),wordWrap='LTR',alignment=align,
        spaceAfter=0,allowWidows=0,allowOrphans=0)
    p=Paragraph(s,st); pw,ph=p.wrap(w,900)
    if maxh is not None and ph>maxh+.1: raise ValueError(f'Page {PAGE} text overflow {ph}>{maxh}: {s[:70]}')
    if y+ph>H-39: raise ValueError(f'Page {PAGE} footer collision {y+ph}: {s[:70]}')
    p.drawOn(C,x,H-y-ph)
    LAYOUT.append({'page':PAGE,'x':x,'y':y,'w':w,'h':ph,'text':s})
    return ph

def rect(x,y,w,h,fill=WHITE,stroke=LINE,r=7,dash=False,lw=.8):
    C.saveState(); C.setFillColor(col(fill)); C.setStrokeColor(col(stroke)); C.setLineWidth(lw)
    if dash: C.setDash(3,3)
    C.roundRect(x,H-y-h,w,h,r,stroke=1,fill=1); C.restoreState()

def box(x,y,w,h,title,subtitle='',color=TEAL,fill=WHITE,size=11,dash=False):
    rect(x,y,w,h,fill,color,dash=dash)
    body=f'<b>{title}</b><br/>{subtitle}' if subtitle else title
    current=size
    while True:
        st=ParagraphStyle('b',fontName='CN',fontSize=current,leading=current*1.22,
            textColor=col(color),wordWrap='LTR',alignment=1)
        p=Paragraph(body,st);_,ph=p.wrap(w-10,500)
        if ph<=h-6: break
        current-=.2
        if current<7.6: raise ValueError(f'Box overflow p{PAGE}: {title} / {subtitle} {ph}/{h}')
    p.drawOn(C,x+5,H-y-(h+ph)/2)

def arrow(points,color=MUTED,dash=False,width=1.1,head=5):
    C.saveState();C.setStrokeColor(col(color));C.setFillColor(col(color));C.setLineWidth(width)
    if dash:C.setDash(3,2)
    p=C.beginPath();p.moveTo(points[0][0],H-points[0][1])
    for x,y in points[1:]:p.lineTo(x,H-y)
    C.drawPath(p);C.setDash()
    x,y=points[-1];x0,y0=points[-2];a=math.atan2(y-y0,x-x0)
    p=C.beginPath();p.moveTo(x,H-y)
    for d in [-.5,.5]:p.lineTo(x-head*math.cos(a+d),H-(y-head*math.sin(a+d)))
    p.close();C.drawPath(p,fill=1,stroke=0);C.restoreState()

def rule(x,y,w,color=LINE):
    C.setStrokeColor(col(color));C.setLineWidth(.7);C.line(x,H-y,x+w,H-y)

def page(title,sub='',chapter='Algorithm guide',first=False):
    global PAGE
    if PAGE:C.showPage()
    PAGE+=1
    C.bookmarkPage(f'p{PAGE}');C.addOutlineEntry(f'{PAGE:02d}  {title}',f'p{PAGE}',level=0)
    C.setFillColor(col(WHITE));C.rect(0,0,W,H,stroke=0,fill=1)
    text(32,18,'M4 / PI OFF',9,TEAL,True)
    C.setFont('CN',8.5);C.setFillColor(col(MUTED));C.drawRightString(W-32,H-26,chapter)
    text(32,38,title,22 if not first else 21,NAVY,True)
    if sub:para(33,70,W-66,sub,9.4,13.4,MUTED)
    rule(32,H-32,W-64)
    text(32,H-24,'CONWIP intelligent control · Implementation-checked edition · 2026.10',8,MUTED)
    C.setFont('CN',8);C.drawRightString(W-32,15,f'{PAGE:02d} / 20')

def section(x,y,w,title,body,color=TEAL,size=11):
    text(x,y,title,12,color,True,max_width=w)
    ph=para(x,y+23,w,body,size)
    return y+23+ph

def callout(x,y,w,h,title,body,color=TEAL,size=10.3):
    rect(x,y,w,h,PALE,LINE)
    C.setFillColor(col(color));C.rect(x,H-y-h,4,h,stroke=0,fill=1)
    text(x+15,y+12,title,11.3,color,True,max_width=w-30)
    para(x+15,y+34,w-30,body,size,maxh=h-43)

def table(x,y,widths,headers,rows,rowheight=36,size=10):
    style=ParagraphStyle('cell',fontName='CN',fontSize=size,leading=size*1.4,
        wordWrap='LTR',textColor=col(INK))
    hstyle=ParagraphStyle('th',parent=style,fontName='CNB',textColor=col(WHITE))
    cells=[[Paragraph(str(t),hstyle) for t in headers]]+[[Paragraph(str(t),style) for t in r] for r in rows]
    t=Table(cells,colWidths=widths,rowHeights=[34]+[rowheight]*len(rows))
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),col(NAVY)),('ROWBACKGROUNDS',(0,1),(-1,-1),[col(WHITE),col(PALE)]),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),8),
        ('LINEBELOW',(0,0),(-1,0),.6,col(NAVY)),('LINEBELOW',(0,1),(-1,-1),.4,col(LINE))]))
    tw,th=t.wrap(sum(widths),800)
    if y+th>H-39:raise ValueError(f'Table footer collision p{PAGE}')
    t.drawOn(C,x,H-y-th)
    return th

def evidence(key,x=438,y=398,w=369,h=130):
    r=summary[key,'q_ensemble_5'];d=base['reward_x1000_mean']-r['reward_x1000_mean']
    callout(x,y,w,h,'Ablation evidence: retaining this module improves reward',
       f"""Without module, Reward ×1000: <b>{r["reward_x1000_mean"]:.3f}</b>; full version: <b>{base["reward_x1000_mean"]:.3f}</b>."""
       f"""Mean advantage of retaining the module: <b>{d:.3f}</b>. Conditional on other modules and five-model ensembling; not its standalone benefit.""",TEAL,10.2)

# 01: Online networks, target computation, replay and deployment.
page('M4 Independent Q Ensemble: Algorithm Overview','Global features + GRU + QMIX + 3-step + PER; five independently trained models; PI residual guidance off.',first=True)
rect(24,92,794,139,WHITE,NAVY,dash=True)
text(36,101,'A  Online networks and centralized training / independent member k, k = 1…5',10.6,NAVY,True)
box(38,137,105,48,'3 unit histories H(t)','Each: 4 frames × (45+24)',TEAL,size=9.4)
box(162,137,105,48,'Shared GRUθ','69 → 64; encode each unit',TEAL,size=9.4)
box(286,126,110,69,'3 separate Q heads','A / B / assembly<br/>3 action Q values each',TEAL,size=9.5)
box(415,137,99,48,'Gather actual-action Q','qA(aA), qB(aB), qC(aC)',BLUE,size=9)
box(537,137,112,48,'QMIXθ → Qtotal','g(t) conditions monotonic weights',BLUE,size=9.4)
box(682,130,121,65,'TD error and loss','δ = Qtotal − y<br/>w × Huber(δ)',ORANGE,size=10)
for a,b in [(143,162),(267,286),(396,415),(514,537),(649,682)]:arrow([(a,161),(b,161)])
box(38,200,130,22,'g(t): 24-D global summary',color=BLUE,size=8.7)
arrow([(168,211),(592,211),(592,185)],BLUE)
text(291,201,'Legal-action mask; replay supplies a(t)',8.3,MUTED)
rect(24,241,794,130,WHITE,NAVY,dash=True)
text(36,250,'B  Target computation: online selection, target evaluation / Double DQN',10.4,NAVY,True)
box(38,282,101,57,'Successor history H(t+n)','n ≤ 3; usually n = 3',TEAL,size=9.1)
box(158,274,119,32,'Online Qθ: legal argmax',color=TEAL,size=9.1)
box(158,319,119,32,'Target Qθ−: output Q',color=TEAL,size=9.1,dash=True)
arrow([(139,310),(148,310),(148,290),(158,290)])
arrow([(148,310),(148,335),(158,335)])
box(311,290,114,53,'Gather target Q at online action','q−i(a*i)',BLUE,size=9.1,dash=True)
arrow([(277,290),(300,290),(300,304),(311,304)])
arrow([(277,335),(300,335),(300,330),(311,330)])
box(458,290,130,53,'Target QMIXθ−','g(t+n) → target Qtotal',BLUE,size=9.1,dash=True)
arrow([(425,317),(458,317)])
box(626,281,177,69,'TD target y','R(n) + γⁿ(1−d) Qtotal−<br/>No terminal bootstrap; γ = 0.99',ORANGE,size=9.6,dash=True)
arrow([(588,317),(626,317)])
arrow([(778,281),(809,281),(809,212),(743,212),(743,195)],ORANGE)
text(40,356,'Copy online GRU, Q heads and QMIX to targets every 250 gradient updates. No gradient flows through targets.',8.7,MUTED)
rect(24,383,794,68,PALE,LINE,dash=True)
text(36,391,'C  Experience and learning',9.8,ORANGE,True)
box(145,398,112,39,'Environment trajectory',color=ORANGE,size=9.4)
box(279,398,125,39,'3-step return R(n)',color=ORANGE,size=9.4)
box(428,398,129,39,'PER storage and sampling',color=ORANGE,size=9.4)
box(581,398,222,39,'Samples → A / B; |δ| updates priorities',color=ORANGE,size=9.2)
for a,b in [(257,279),(404,428),(557,581)]:arrow([(a,418),(b,418)],ORANGE)
rect(24,462,794,73,WHITE,PURPLE,dash=True)
text(36,471,'D  Deployment',10,PURPLE,True)
box(145,480,188,40,'Same observations → 5 independent models',color=PURPLE,size=9.6)
box(357,480,177,40,'Average Q per unit and action',color=PURPLE,size=9.6)
box(558,480,245,40,'Legal argmax → [decrease / hold / increase]',color=PURPLE,size=9.5)
arrow([(333,500),(357,500)],PURPLE);arrow([(534,500),(558,500)],PURPLE)
text(37,520,'Deployment uses no QMIX, target networks, replay or PI, but still broadcasts global features. Solid: online; small dashed boxes: targets.',8.5,MUTED)

# 02
page('Network Components and Training Roles','Online networks, target networks and ensemble members have distinct roles in M4.','Reading map')
section(34,107,366,'Controller overview',
    'Upgrade three workshop supervisors into a team: <b>read a plant-wide dashboard, remember recent changes, and learn from plant-wide performance</b>. During training, look several steps ahead and revisit difficult experiences. At deployment, five independently trained teams score each action.')
table(34,225,[113,253],['Diagram component','Meaning in this project'],[
    ['3 agents','Control CONWIP caps for A processing, B processing and assembly.'],
    ['5 independent models','Five training runs of the same algorithm, not five production units.'],
    ['Online / target networks','A learner and its delayed copy within one model, not ensemble members.'],
    ['PI OFF','No PI advice input, residual decoding or PI-guided exploration.'],
],rowheight=56,size=10)
section(439,107,367,'Key differences from the MATD3 reference',
    'MATD3 has actors, twin critics and a minimum over two target values. Here, <b>discrete-action value learning</b> scores three candidate actions, and QMIX learns a team value.<br/><br/>There are no actors, twin-critic minimum or target-action noise. Double-DQN-style targets use online action selection and target-network evaluation.')
callout(439,291,367,108,'Two different meanings of centralization',
    'QMIX is used only during training, but the 24-D global summary is also broadcast during execution. Execution is therefore not strictly local and communication-free.',BLUE)
section(439,421,367,'Suggested reading order',
    'Pages 3-4: problem and objective. Pages 5-13: individual improvements. Pages 14-16: learning and control end to end. Pages 17-19: what the experiments establish. Page 20: glossary and evidence.',size=10.6)

# 03
page('Production: Two Part Lines Must Deliver a Matched Pair','The controller changes permitted work-in-process in each unit, not machine speed.','01 / Environment and problem')
box(36,125,185,63,'Part A processing line','5 stages, 1 machine per stage',TEAL)
box(36,225,185,63,'Part B processing line','5 stages, 1 machine per stage',BLUE)
box(258,125,123,63,'A finished-parts buffer','Capacity: 3 parts',TEAL)
box(258,225,123,63,'B finished-parts buffer','Capacity: 3 parts',BLUE)
box(432,164,186,75,'Pair parts for assembly','1 A + 1 B<br/>3 stages, 2 parallel machines each',PURPLE)
box(663,175,141,54,'Completed product sets','Measure rolling throughput',ORANGE)
arrow([(221,156),(258,156)],TEAL);arrow([(221,256),(258,256)],BLUE)
arrow([(381,156),(405,156),(405,186),(432,186)],TEAL)
arrow([(381,256),(405,256),(405,218),(432,218)],BLUE)
arrow([(618,202),(663,202)],PURPLE)
text(35,306,'Like making burgers: plenty of buns cannot replace missing patties.',13,NAVY,True)
section(35,344,365,'Where does the difficulty come from?',
    'A/B processing times vary randomly, throughput targets change, and individual stages temporarily slow down. Releasing more WIP can feed a starved downstream unit or fill an already crowded buffer. <b>A busier local line does not guarantee a better plant.</b><br/><br/>There is no separate random breakdown/repair module. Disturbances come from processing-time randomness, target changes and processing-mean changes.',size=10.6)
section(439,306,367,'Simulation time scale',
    'Warm up for 720 minutes, then control for 3600 minutes. Decisions occur every 10 minutes: 360 per episode. Throughput and reference use 120-minute rolling windows. Initial caps: A=3, B=3, assembly=2.',size=10.6)
callout(439,411,367,125,'Six consecutive 600-minute regimes',
    'Targets: 3 → 6 → 5 → 5 → 5 → 3 sets/hour. Regime 3 increases A/B variability; regime 4 slows B stage 3 by 15%; regime 5 instead slows A stage 2 by 15%; the last restores normal conditions.',ORANGE,10.1)

# 04
page('Good Control Is More Than Minimizing Throughput Error','Tracking, inventory and cap adjustments share one reward.','01 / Environment and problem')
for x,w,t,s,c in [(35,240,'Throughput tracking error','Larger deviation from the rolling target costs more',TEAL),(300,240,'Work-in-process inventory','More work tied up in production costs more',BLUE),(565,240,'Cap-adjustment magnitude','Larger and more frequent changes cost more',ORANGE)]:box(x,113,w,71,t,s,c,size=11)
text(283,142,'+',12,MUTED);text(548,142,'+',12,MUTED)
rect(35,205,770,63,PALE,LINE)
para(52,218,739,'r(t) = − [ tracking cost + 0.05 × mean WIP / 38 + 0.005 × Σ|cap change| / 3 ]',12.2,19,NAVY,align=1)
section(35,294,365,'Why are rewards usually negative?',
    "Reward is negative total cost, so <b>−35 is better than −39</b>. Reward ×1000 is the episode's mean per-step reward multiplied by 1000, then averaged across test episodes. It is neither cumulative return nor exploratory training reward.<br/><br/>Tracking cost is squared normalized error. Episode NRMSE is the square root of mean tracking cost and excludes inventory and adjustment costs.",size=10.7)
section(439,294,367,'From original MADQN to new M4',
    'The matched MADQN baseline uses three feedforward Q networks, current 45-D observations, independent learning, 1-step targets and uniform replay. New M4 adds spatial coordination, short-term memory, team-value learning and training enhancements, then ensembles five models.<br/><br/>The contribution is <b>environment-specific integration and validation</b>. GRU, QMIX and PER themselves are established methods, not inventions of this project.',size=10.7)
para(36,507,768,'WIP accounting: A WIP + B WIP + 2 × assembly WIP + both finished-parts buffers. Each assembly set contains two parts.',9.2,14,MUTED)

# 05
page('Global Features: A Plant-Wide Dashboard for Every Supervisor','Address the blind spot where one line looks healthy but the bottleneck is elsewhere.','02 / Improvement 1: wider visibility')
table(35,112,[125,244],['The 24-D summary','Contents'],[
    ['6-D: unit states','WIP and cap for A, B and assembly.'],
    ['6-D: shared production information','A/B buffers, throughput/reference, relative error, current target and window target.'],
    ['12-D: operating summaries','Busy, blocked, idle and pair-starvation summaries for each unit.'],
],rowheight=66,size=10.2)
box(35,365,148,59,'Original local observation','45 dimensions',TEAL)
box(206,365,198,59,'Append global summary','45 + 24 = 69 dimensions',BLUE)
arrow([(183,395),(206,395)])
para(35,442,369,'These features summarize three observable local vectors; they do not reveal the future. The original 45-D observation already includes shared buffers and throughput, so removing global features does not eliminate every plant-wide signal.',10.6)
section(439,112,367,'Why does this help this production system?',
    "Suppose A's buffer is almost full, B's almost empty, and assembly waits for a pair. An A supervisor focused only on local utilization may release more work and worsen accumulation. Seeing B and assembly makes plant-wide balancing possible.")
section(439,251,367,'Characteristics, benefits and costs',
    'The summary links the three units spatially so Q networks can learn who is waiting for whom. It requires timely broadcast during execution. The ablation removes only these 24 dimensions from Q-network inputs; <b>QMIX retains its 24-D conditioning input</b>.',size=10.7)
evidence('no_global',y=397,h=136)

# 06
page('GRU: Watch a Short Video, Not Just a Photograph','GRU stands for Gated Recurrent Unit.','03 / Improvement 2: remember changes')
text(35,112,'The same buffer count of 2 can describe two different situations',13,NAVY,True)
for row,(label,vals,color) in enumerate([('Accumulating',[0,1,2,2],ORANGE),('Being consumed',[3,3,2,2],TEAL)]):
    y=159+row*90;text(35,y+20,label,11.5,color,True)
    for j,v in enumerate(vals):
        box(148+j*82,y,62,57,str(v),['t−3','t−2','t−1','t'][j],color,size=12)
        if j<3:arrow([(210+j*82,y+28),(230+j*82,y+28)],color)
para(490,161,311,'The final frame is identical but the trends differ. A current-frame network cannot easily identify the path taken. GRU compresses recent frames into short-term memory, giving action scoring information about direction of change.',11.1)
para(490,266,311,'Illustration only. Actual actions also depend on the target, pair starvation and plant-wide state. These counts alone do not determine an action.',10.2,color=MUTED)
section(35,372,365,'Gates do not retain all past information equally',
    'The <b>update gate</b> controls how much old memory remains. The <b>reset gate</b> controls how much old information contributes to a candidate memory. Noise can be downweighted while persistent blocking or target changes are retained. Gate values are learned, not hand-coded thresholds.',size=10.7)
section(439,372,367,'Improvement over original MADQN',
    'The original feedforward network scores the current observation directly. GRU first interprets a short sequence. The same error may mean a new deviation or an ongoing recovery. Memory can distinguish these and reduce reactions to momentary fluctuations.',size=10.7)

# 07
page('GRU in Code: Shared Encoder, Distinct Roles','Each forward pass reads the latest 4 frames; hidden state does not persist across the whole episode.','03 / Improvement 2: architecture')
for i,(t,c) in enumerate([('A history',TEAL),('B history',BLUE),('Assembly history',PURPLE)]):
    box(35,127+i*85,119,55,t,'4 × 69',c)
    arrow([(154,154+i*85),(196,154+i*85)],c)
rect(196,118,180,243,PALE,TEAL)
text(215,135,'Shared GRU weights',12,TEAL,True,max_width=143)
para(214,170,143,'Each unit encodes its own time sequence. Weights are shared; hidden representations are not concatenated across units.',10.7)
para(214,275,143,'Each forward pass starts with zero hidden state and uses the final 64-D representation.',10.2,color=MUTED)
for i,(t,c) in enumerate([('A-specific Q head',TEAL),('B-specific Q head',BLUE),('Assembly-specific Q head',PURPLE)]):
    box(418,127+i*85,174,55,t,'64 → 64 → 64 → 3',c)
    box(635,127+i*85,169,55,'Decrease · Hold · Increase','3 Q values + legal-action mask',c,size=10)
    arrow([(376,154+i*85),(418,154+i*85)],c);arrow([(592,154+i*85),(635,154+i*85)],c)
section(35,388,367,'What is shared, and what is independent?',
    'Within one model, GRU weights are shared but the three output heads have separate parameters to preserve unit roles. Across five models, GRUs, heads, QMIX, optimizers and replay buffers are independent.<br/><br/>Four frames cover t−3 through t, spanning 30 minutes. At episode start, repeat the initial frame until the history is full.',size=10.3)
evidence('no_gru',y=388,h=146)

# 08
page('QMIX: Combine Three Local Scores into a Team Score',"A shared reward reports plant performance; QMIX also learns how the agents' values combine.",'04 / Improvement 3: coordination')
section(35,112,367,'The difficulty with independent learning',
    "If completed-product throughput falls, A, B and assembly all receive the same poor reward. Should B's cap rise or A's accumulation shrink? Independent updates from a shared reward make it harder to connect an agent's action with its teammates.<br/><br/>QMIX combines the three actual-action Q values, conditioned on the global summary, into team value Qtotal.")
section(35,318,367,'Like a coach reviewing a relay race',
    'The coach examines handoffs, not just individual running speed. Mixer loss backpropagates through all three Q heads and the shared GRU, aligning local action scores with the shared objective.<br/><br/>QMIX does not explicitly calculate contribution percentages or manually assign responsibility.',size=10.7)
for i,(label,color) in enumerate([("qA: A's actual-action value",TEAL),("qB: B's actual-action value",BLUE),("qC: assembly's actual-action value",PURPLE)]):
    box(441,118+i*69,199,43,label,color=color,size=10.4)
    arrow([(640,139+i*69),(663,139+i*69),(663,230),(684,230)],color)
box(684,177,121,108,'QMIX','g(t) conditions mixing<br/>Output: Qtotal',BLUE,size=12)
box(441,341,199,58,'Global summary g(t)','24-D, not the raw 135-D global state',BLUE,size=10)
arrow([(640,370),(744,370),(744,285)],BLUE)
callout(441,427,364,107,'Local Q values are internal team scores',
    'They need not represent physical throughput or separate unit profits. The main training constraint acts on the mixed team value. Do not interpret the three Q values as true contribution shares.',BLUE,10.2)

# 09
page('QMIX Monotonicity: Why Local High Scores Matter','For a fixed state, increasing any local Q cannot decrease the mixed team Q.','04 / Improvement 3: architecture')
rect(35,112,770,77,PALE,LINE)
para(54,126,732,'Qtotal = f(qA, qB, qC ; g), with ∂Qtotal / ∂qi ≥ 0',17,24,BLUE,align=1)
para(54,159,732,"Implementation: a 24-D global summary generates two mixing layers' weights. Absolute values enforce nonnegative weights; the hidden layer uses ELU and width 32.",10.2,15,MUTED,align=1)
section(35,217,366,'What does the constraint enable?',
    'Imagine three knobs connected to one dashboard. Turning any knob toward a higher local score cannot lower the team score. With fixed inputs and separable legal-action constraints, independent local argmax choices maximize the mixed value.<br/><br/>Training can evaluate the team while execution avoids enumerating all joint actions through QMIX. See [1].',size=10.7)
section(35,430,366,'Not a universal team model',
    'Monotonicity limits expressiveness. Arbitrary nonmonotonic coordination relationships, where a locally preferred action must be sacrificed, cannot always be represented exactly.',size=10.4)
section(439,217,367,"Describe this project's execution precisely",
    'Deployment does not run the mixer, but each Q network still reads the broadcast global summary. Averaging local Q values across models adds another operation. Single-model monotonicity <b>does not guarantee that the ensemble action maximizes the mean of five Qtotal values</b>. Ensemble benefits are empirical.',size=10.7)
evidence('no_qmix',y=398,h=136)

# 10
page('3-Step: Do Not Judge a Release Before Its Effects Arrive','n-step does not peek into the future. It waits for actual transitions, then learns from several consequences together.','05 / Improvement 4: longer learning horizon')
for j,(t,s) in enumerate([('t','Adjust cap'),('t+1','Parts keep moving'),('t+2','Consequences emerge'),('t+3','Bootstrap from successor value')]):
    x=35+j*199;box(x,122,173,69,t,s,ORANGE,size=12)
    if j<3:arrow([(x+173,157),(x+199,157)],ORANGE)
rect(35,218,770,65,PALE,LINE)
para(48,229,744,'y = r(t) + γ r(t+1) + γ² r(t+2) + γ³ (1−d) Qtotal−(t+3)',14,21,ORANGE,align=1)
para(50,259,741,'Usually n = 3. Near episode end use the actual remaining n, with no bootstrap after termination. γ = 0.99.',10,15,MUTED,align=1)
section(35,309,366,'Why look several steps ahead in production?',
    'After a cap increase, extra parts are still in transit and throughput may not rise immediately. A one-step target may initially see only inventory cost. Three-step returns bring rewards from three actual decision intervals back to the current action.<br/><br/>Three steps cover about 30 minutes, not every processing or rolling-window delay. Longer effects still rely on value bootstrapping.',size=10.7)
section(439,309,367,'Numerical illustration (not experimental data)',
    'For rewards −0.04, −0.03 and −0.02, the discounted short return is approximately −0.0893, plus 0.99³ times the successor value. This extends the learning horizon rather than tripling reward. Longer returns can increase variance and off-policy bias; larger n is not always better.',size=10.4)
evidence('one_step',y=427,h=108)

# 11
page('PER: An Updating Notebook of Prediction Mistakes','Prioritized Experience Replay assigns priority according to prediction error.','06 / Improvement 5: targeted replay')
for x,w,t,s in [(35,160,'Replay buffer','Capacity: 100,000 samples'),(236,172,'Priority-based sampling','Batch: 64, with replacement'),(450,172,'Update network','Weighted Huber loss'),(663,142,'Update priorities','Batch |TD error| + 10<super>−5</super>')]:box(x,120,w,68,t,s,ORANGE,size=10.7)
for a,b in [(195,236),(408,450),(622,663)]:arrow([(a,154),(b,154)],ORANGE)
arrow([(734,188),(734,218),(115,218),(115,188)],ORANGE)
text(313,225,"After learning, update each sample's difficulty tag",10.3,ORANGE)
section(35,267,367,'Difficult does not mean lowest reward',
    'A bad outcome can have low TD error if already predicted accurately. An unexpectedly good outcome can have high TD error. PER favors <b>experiences the current model cannot yet explain</b>, not only failures.<br/><br/>Target switches or congestion turning points may therefore receive more replay. The code does not hand-assign priority to regime changes.',size=10.5)
section(439,267,367,'How does PER avoid focusing only on mistakes?',
    'Sampling probability is proportional to priority raised to 0.6, moderating concentration. Importance weights reduce nonuniform-sampling bias; β rises from 0.4 to 1 and weights are batch-normalized. New experiences receive the current maximum priority.<br/>See [2].',size=10.4)
callout(35,452,369,82,'Different from n-step',
    "n-step sets how far one sample's target looks ahead. PER sets how often that sample is revisited. They address different issues.",ORANGE,10)
evidence('uniform_replay',y=423,h=111)

# 12
page('Five-Model Independent Q Ensemble: Average Scores, Then Decide','Not majority voting, action averaging, or averaging five separate rewards and calling that deployed performance.','07 / Improvement 6: multiple learned views')
section(35,112,366,'Five evaluators, each fully trained',
    'The five members share an architecture but independently initialize, explore and learn under different seeds. Each has its own replay buffer, optimizer, targets and best validation checkpoint.<br/><br/>At execution, all receive the same history. Average Q arithmetically for each unit and candidate action, then select the highest-scoring legal action.')
rect(35,322,366,70,PALE,LINE)
para(45,335,346,'Mean qi(a) = [q¹i(a) + … + q⁵i(a)] / 5<br/>ai = argmax of mean qi over legal actions',12,20,PURPLE,align=1)
callout(35,414,366,120,'Observed change in this test',
    f"""New M4 single-model mean: {summary["no_pi","single"]["reward_x1000_mean"]:.3f}; five-model joint deployment: {base["reward_x1000_mean"]:.3f}. Mean NRMSE falls from 0.13171 to 0.12302. These compare actual control trajectories.""",PURPLE,10.2)
text(439,112,'Why is it not voting?',12,PURPLE,True)
table(439,145,[74,79,79,95],['Illustrative member','Decrease Q','Hold Q','Increase Q'],[
    ['1','−1.0','−1.2','−2.0'],['2','−1.0','−1.2','−2.0'],['3','−1.0','−1.2','−2.0'],
    ['4','−4.0','−0.8','−2.0'],['5','−4.0','−0.8','−2.0'],['Mean','−2.20','−1.04','−2.00'],
],rowheight=31,size=10)
para(439,380,367,'Teaching numbers, not model outputs. Voting chooses decrease by 3:2. Mean-Q selection chooses hold because −1.04 is highest. Score margins also affect the decision.',10.5)
para(439,455,367,"Benefit: may reduce accidental single-run bias. Cost: roughly five networks' training and inference. Q scales are not explicitly calibrated, and errors are not guaranteed to cancel.",10.3,color=MUTED)

# 13
page('With PI Off, Networks Directly Choose Cap Changes','M4 remains the project version name; this does not imply retaining the old PI branch.','08 / Version scope and execution')
table(35,112,[151,291,328],['Aspect','Earlier PI-residual variant','Current version: PI OFF'],[
    ['Input','69-D features + 3-D PI action encoding.','69-D; no PI advice appended.'],
    ['Output meaning','Correction around the PI proposal.','Direct decrease, hold or increase.'],
    ['Exploration / initialization','May use PI guidance and a zero-residual prior.','No PI guidance or PI-specific hold bias.'],
    ['Boundary protection','Check legality after residual mapping.','Mask illegal actions using current caps directly.'],
],rowheight=43,size=10.1)
box(35,346,170,64,'Current caps','A = 3, B = 0, assembly = 8',TEAL,size=10.7)
box(255,346,232,64,'Filter legal actions','B cannot decrease; assembly cannot increase',BLUE,size=10.7)
box(537,346,268,64,'Suppose: hold, increase, decrease','New caps = [3, 1, 7]',PURPLE,size=10.7)
arrow([(205,378),(255,378)]);arrow([(487,378),(537,378)])
section(35,441,367,'Action indices and their meaning',
    'Indices 0 / 1 / 2 mean cap changes −1 / 0 / +1. Caps stay within 0-8 and illegal candidates are masked. Reducing a cap does not remove existing WIP; it limits subsequent releases.',size=10.4)
section(439,441,367,'One fewer module and dependency',
    'The new version needs no PI formula, advice or residual constraint. The choice follows existing project results, not a claim that PI is always useless. All following ablations keep PI off.',size=10.4)

# 14
page('Learning from One Batch: Replay to Parameter Update','Expand the training part of the overview and identify what each arrow carries.','09 / One learning update')
steps=[
 ('1','Sample a batch','PER draws 64 experiences: current history, global summary, actual actions, multi-step return, successor history and terminal flag.'),
 ('2','Compute current estimate','Online GRU and Q heads score actions. Gather executed-action Q values and combine them with online QMIX into current Qtotal.'),
 ('3','Compute a stable target','At the successor history, online Q selects the best legal action. Target Q evaluates that action, then target QMIX produces successor value.'),
 ('4','Backpropagate','Multi-step reward plus discounted successor value gives y. Weighted Huber loss between Qtotal and y updates online parameters.'),
 ('5','Maintain the learning system',"Update replay priorities using this batch's |TD error|. Every 250 gradient updates, copy online weights to targets."),
]
for j,(n,t,b) in enumerate(steps):
    y=109+j*82
    box(35,y,38,38,n,color=ORANGE,size=15)
    if j<4:arrow([(54,y+38),(54,y+78)],ORANGE)
    text(94,y+1,t,12,NAVY,True);para(94,y+25,394,b,10.2,15,maxh=50)
rect(531,111,274,242,PALE,LINE)
text(549,128,'Two networks, two responsibilities',12,BLUE,True)
para(549,163,238,'<b>Online network:</b> selects the best successor action and receives gradients for the current estimate.<br/><br/><b>Target network:</b> provides a steadier estimate of that selected action. Target y is detached.<br/><br/>Separating selection from evaluation helps reduce overestimation. See [3].',10.4,16)
callout(531,382,274,153,'Do not draw this as TD3',
    "This is online selection and target evaluation, not the minimum of twin critics. The five members do not average each other's TD targets; they train independently.",BLUE,10.4)

# 15
page('Training, Validation and Test Have Different Jobs','This guide uses completed experiments. No retraining or new seeds; comparison definitions remain consistent.','09 / Training and evaluation')
for x,t,s,c in [(35,'Training','Explore, interact and update parameters',ORANGE),(300,'Validation','No exploration; select checkpoint and stop',BLUE),(565,'Test','No exploration; report fixed-seed performance',TEAL)]:box(x,114,240,68,t,s,c,size=11)
arrow([(275,148),(300,148)]);arrow([(540,148),(565,148)])
para(36,190,769,'Validation seeds: 16101-16112. Test seeds: 30101-30130. Validation selects models; test compares them. Do not interchange the two.',9.1,14,MUTED)
table(35,212,[145,225],['Training setting','Current implementation'],[
    ['Optimizer / learning rate','Adam / 0.0001'],['Learning trigger','After ≥1000 replay samples, update every 4 interactions'],
    ['Batch / gradient clipping','64 / norm limit 10'],['Discount / return length','0.99 / 3 (shorter at episode end)'],
    ['Exploration ε','1 → 0.05 over the first 108,000 steps'],['Target synchronization','Hard copy every 250 gradient updates'],
],rowheight=39,size=10)
section(439,212,367,'How is the final model selected?',
    'Training seeds: 13101-13105. Validate on 12 fixed seeds every 50 episodes. Reward gain >0.05 (×1000 units) is an improvement. If reward differs by at most 0.05, an NRMSE decrease >0.0002 can update the best point.<br/><br/>Train at least 300 episodes; stop after 4 validations without improvement, with a 1250-episode limit. Use the best validation checkpoint, not the final episode.',size=10.5)
callout(439,420,367,115,'Early stopping is not complete mathematical convergence',
    'Across six configurations and 30 members, 27 meet validation early stopping and 3 reach the budget limit. New M4, no-GRU and no-QMIX each have one limit-reaching member. Do not claim every model fully converged.',BLUE,10.1)

# 16
page('Putting the Modules Together: B Suddenly Slows Down','A teaching scenario showing module roles, not a timestamp-by-timestamp explanation extracted from a test trajectory.','10 / A production-floor story')
items=[
 ('01','Identify that not every unit is slow','Global features','Combine A/B buffers, unit WIP and operating states. The model may recognize A accumulation, B undersupply and assembly waiting for matching parts.',TEAL),
 ('02','Distinguish a trend from a fluctuation','GRU','Review four frames: is B starvation persistent and A accumulation growing? The same present error can have different histories.',TEAL),
 ('03','Coordinate for completed sets','QMIX','Training evaluates the joint actions with team value, allowing local scores to favor plant-wide balance instead of local utilization alone.',BLUE),
 ('04','Wait for downstream consequences','3-step','Backpropagate actual rewards from several intervals so release decisions learn some delayed consequences sooner.',ORANGE),
 ('05','Revisit poorly understood turning points','PER','A large target-prediction gap increases the chance of replay. Priority changes again as prediction improves.',ORANGE),
 ('06','Combine independent training perspectives','Q ensemble','At execution, five models score actions separately. Average the scores to choose the three unit actions, reducing accidental single-run preferences.',PURPLE),
]
for j,(n,t,m,b,c) in enumerate(items):
    y=110+j*69
    box(35,y,40,39,n,color=c,size=13)
    text(92,y,t,11.5,c,True);text(652,y,m,10.5,c,True)
    para(92,y+23,708,b,10.1,15,maxh=37)

# 17
page('Ablations: How Much Reward Does Each Module Add?','PI stays off. Remove one module at a time and hold other settings fixed. All results here are test performance of five-model ensembles.','11 / Ablation evidence')
order=[('no_gru','GRU'),('no_global','Global features'),('no_qmix','QMIX'),('one_step','3-step'),('uniform_replay','PER')]
text(36,111,'Reward advantage of retaining a module (full − ablated)',12,NAVY,True)
x0=178;x1=757;y0=160;scale=(x1-x0)/5.2
for j in range(6):
    x=x0+j*scale;C.setStrokeColor(col(LINE));C.setLineWidth(.5);C.line(x,H-146,x,H-417)
    text(x-3,425,str(j),9,MUTED)
for j,(key,label) in enumerate(order):
    d=contrast[key,'q_ensemble_5','reward_x1000'];v=-d['delta_ablation_minus_baseline'];lo=-d['ci95_high'];hi=-d['ci95_low'];y=y0+j*55
    text(35,y-7,label,11,INK,True)
    C.setStrokeColor(col(TEAL));C.setLineWidth(2);C.line(x0+lo*scale,H-y,x0+hi*scale,H-y)
    for z in [lo,hi]:C.line(x0+z*scale,H-y-4,x0+z*scale,H-y+4)
    C.setFillColor(col(TEAL));C.circle(x0+v*scale,H-y,4,stroke=0,fill=1)
    text(x0+v*scale+9,y-17,f'{v:.3f}',9.8,TEAL,True)
    text(35,y+11,f'95% CI [{lo:.3f}, {hi:.3f}]',8.2,MUTED)
callout(35,459,370,76,'What does this figure support?',
    'All five reward-difference intervals lie above zero. On this fixed test set, retaining each module improves mean reward.',TEAL,9.8)
callout(435,459,370,76,'What does it not support?',
    'Do not add the five differences or claim statistically different contributions merely by ranking their means.',ORANGE,9.8)
para(180,443,596,'Horizontal bars: 95% intervals from 8,000 paired bootstrap resamples over shared test seeds; no multiplicity correction.',8.5,12,MUTED)

# 18
page('Better Reward Does Not Mean Every Metric Decreases','Aggregate reward trades off tracking, WIP and adjustment cost. NRMSE is not the only outcome.','11 / Ablation evidence')
labels={'no_pi':'New M4 (all modules)','no_global':'Without global features','no_gru':'Without GRU','no_qmix':'Without QMIX','one_step':'3-step → 1-step','uniform_replay':'PER → uniform replay'}
rows=[]
for key in labels:
    r=summary[key,'q_ensemble_5']
    rows.append([labels[key],f'{r["reward_x1000_mean"]:.3f}',f'{r["nrmse_mean"]:.5f}',f'{r["mean_wip_mean"]:.3f}',f'{r["cap_adjustment_points_mean"]:.1f}'])
table(35,112,[224,128,118,135,165],['Five-model strategy','Reward ×1000 ↑','NRMSE ↓','Mean WIP ↓','Cap-change points ↓'],rows,rowheight=37,size=10.3)
section(35,400,366,'GRU has the clearest tracking contribution',
    'Without GRU, NRMSE rises from 0.12302 to 0.13515, approximately +0.01214, with a paired 95% interval excluding zero. The other four NRMSE intervals include zero, so they should not all be described as significant tracking improvements.',size=10.4)
section(439,400,367,'Global features and PER affect more than tracking',
    'Removing either slightly lowers mean NRMSE but increases inventory and adjustment points, worsening reward. This is consistent with improved cost trade-offs, but without reward-component decomposition it does not establish exact attribution.',size=10.4)
para(35,523,770,'Every row averages the same 30 test seeds. A change point is a decision where at least one cap changes, not Σ|cap change|.',8.8,13,MUTED)

# 19
page('Ablation Design and Evidence Limits','Mechanisms explain why a module may help. Ablations show what happens when it is removed from this configuration and test.','12 / Scope of conclusions')
section(35,112,366,'Leave-one-out experiments around new M4',
    'The baseline includes global features, GRU, QMIX, 3-step, PER and five-model ensembling. Each ablation removes one component; PI is always off.<br/><br/>These are <b>conditional contributions</b>: GRU helps when the other modules remain. They are not results for adding GRU alone to original MADQN, nor a complete interaction study.')
section(35,316,366,'What exactly changes when a module is removed?',
    'No-GRU uses a current-frame feedforward network, not a parameter-count-matched substitute. No-QMIX uses separate per-agent TD learning.<br/><br/>Results compare these specific implementations, not abstract effects independent of capacity and learning targets.',size=10.7)
callout(35,471,366,64,'No new-seed confirmation in this round',
    'Uses the fixed training and test seeds.',TEAL,9.7)
section(439,112,367,'Single-model and ensemble evidence are different',
    'Single-model summaries use 5 training members × 30 test seeds. An ensemble is one fixed team of those members, run on 30 seeds.<br/><br/>For single-model reward differences, global-feature and PER intervals include zero; GRU, QMIX and 3-step intervals exclude zero. Clear ensemble gains do not imply every single model benefits reliably.',size=10.7)
section(439,333,367,'Summary of supported findings',
    'Under the specified environment and fixed training/test seeds, the PI-off M4 five-model independent Q ensemble achieves higher aggregate reward. Leave-one-module-out experiments show positive conditional contributions of all five modules to ensemble reward, with GRU providing the clearest tracking-accuracy benefit.',size=10.6)
para(439,467,367,'Test seeds have been used for version comparison, so this is not a fresh external final test. Intervals do not cover all uncertainty from environment changes or retraining multiple ensembles.',9.6,color=MUTED)

# 20
page('The Entire Algorithm on One Memory Card','See more → remember changes → coordinate → learn farther ahead → replay difficulties → combine models.','13 / Glossary and evidence')
table(35,111,[95,272],['Term','Remember this'],[
    ['Global features','Let every unit see what the other two are doing.'],['GRU','Compress four frames into selective short-term memory.'],
    ['QMIX','Mix local action scores into a team value during training.'],['3-step','Connect three real rewards to a successor-value estimate.'],
    ['PER','Replay experiences the model has not yet learned accurately.'],['Independent Q ensemble','Five models score the same action; average before selecting.'],
    ['PI OFF','Choose directly, without PI advice or residuals.'],
],rowheight=42,size=10.1)
para(35,460,367,'<b>Key distinction:</b> ensemble members are independent. Within each member, one GRU is shared and three dedicated Q heads remain. QMIX is training-only; global summaries are also used during execution.',10.5)
section(439,111,367,'Project evidence used in this guide',
    'Architecture/training: run_controlled_m4_ablation.py. Networks, global summary and replay: run_madqn_ablation.py. Environment: pair_v2.py / baseline_algorithms.py.<br/><br/>Values come directly from m4_no_pi_ablation/comparison_summary.json, paired_contrasts.json and training_manifest.json. Only the test partition is used, not validation_audit.',size=9.6)
text(439,287,'Method references (clickable)',11.5,TEAL,True)
refs=[
    ('[1] Rashid et al., 2018: QMIX and monotonic value factorization.','https://arxiv.org/abs/1803.11485'),
    ('[2] Schaul et al., 2016: Prioritized Experience Replay.','https://arxiv.org/abs/1511.05952'),
    ('[3] van Hasselt et al., 2016: Double Q-learning.','https://arxiv.org/abs/1509.06461'),
    ('[4] Cho et al., 2014: early gated recurrent unit architecture.','https://arxiv.org/abs/1406.1078'),
]
for j,(label,url) in enumerate(refs):
    para(439,313+j*39,367,f'<link href="{url}" color="{BLUE}">{escape(label)}</link>',9.6,14)
callout(439,474,367,61,'Status of illustrations and numbers',
    'Code-based diagrams; illustrative examples; real ablations.',TEAL,9.5)

assert PAGE==20
C.save()
(ROOT/'work'/'m4_book_layout.json').write_text(json.dumps(LAYOUT,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'pdf':str(PDF),'pages':PAGE,'paragraphs':len(LAYOUT)},ensure_ascii=False))
