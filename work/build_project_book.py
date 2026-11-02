from pathlib import Path
import csv, json, math
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor, Color, white
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'outputs'/'project_illustrated_book'
OUT.mkdir(parents=True,exist_ok=True)
pdfmetrics.registerFont(TTFont('Body',str(ROOT/'work/fonts/DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('Head',str(ROOT/'work/fonts/DejaVuSans.ttf')))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='Head',italic='Body',boldItalic='Head')
P=OUT/'From_Production_to_M4_Project_Guide.pdf'
c=canvas.Canvas(str(P),pagesize=(595.28,841.89))
c.setTitle('From the Production Floor to M4 Independent Q Ensembling: Illustrated Guide')
c.setAuthor('')
c.setCreator('')
c.setSubject('Paired-assembly CONWIP simulation, historical M4 controllers and experimental results')
W,H=595.28,841.89
L,R=48,547
INK='#183349'; TEAL='#087F8C'; BLUE='#3273AC'; GOLD='#C58935'; MUT='#617487'; LIGHT='#EFF5F8'
page=0;y=0
style=ParagraphStyle('text',fontName='Body',fontSize=11,leading=18,textColor=HexColor(INK),wordWrap='LTR',spaceAfter=0)
small=ParagraphStyle('small',parent=style,fontSize=9,leading=14,textColor=HexColor(MUT))
def text(s,x,top,size=11,color=INK,font='Body'):
    while pdfmetrics.stringWidth(s,font,size)>R-x and size>8:
        size-=.2
    c.setFillColor(HexColor(color));c.setFont(font,size);c.drawString(x,H-top-size,s)
def para(s,size=11,color=INK,gap=8):
    global y
    st=ParagraphStyle('p',parent=style,fontSize=size,leading=size*1.45,textColor=HexColor(color))
    p=Paragraph(s,st);_,h=p.wrap(R-L,1000)
    assert y+h<786,(page,y,h,s[:40])
    p.drawOn(c,L,H-y-h);y+=h+gap
def sub(s):
    global y
    y+=5;text(s,L,y,14,TEAL,'Head');y+=29
def note(s):
    global y
    p=Paragraph(s,small);_,h=p.wrap(R-L-24,1000)
    c.setFillColor(HexColor(LIGHT));c.roundRect(L,H-y-h-20,R-L,h+20,5,fill=1,stroke=0)
    p.drawOn(c,L+12,H-y-h-10);y+=h+32
def start(ch,title,deck):
    global page,y
    if page:c.showPage()
    page+=1;c.setFillColor(HexColor('#FFFFFF'));c.rect(0,0,W,H,fill=1,stroke=0)
    text('CONWIP / Multi-agent production control',L,25,9,MUT,'Head')
    text(f'{page:02d}',R-15,25,9,MUT,'Head')
    c.setStrokeColor(HexColor('#D7E3EA'));c.line(L,H-46,R,H-46)
    text(ch,L,65,10,TEAL,'Head');text(title,L,88,24,INK,'Head')
    y=132;para(deck,10,MUT,17)
    c.setStrokeColor(HexColor('#D7E3EA'));c.line(L,43,R,43)
    text('Illustrated project guide · Existing code and experiments · 2026.10',L,805,8,MUT)
    c.bookmarkPage(f'p{page}');c.addOutlineEntry(title,f'p{page}',0)
def box(x,t,w,h,label,color=TEAL):
    c.setFillColor(HexColor('#F2F7FA'));c.setStrokeColor(HexColor(color));c.roundRect(x,H-t-h,w,h,6,stroke=1,fill=1)
    body=label.replace('|','<br/>');fs=10
    while True:
        st=ParagraphStyle('box',fontName='Head',fontSize=fs,leading=fs*1.22,textColor=HexColor(color),alignment=1)
        p=Paragraph(body,st);_,ph=p.wrap(w-10,500)
        if ph<=h-8:break
        fs-=.2
        assert fs>=7.6,(body,ph,h)
    p.drawOn(c,x+5,H-t-(h+ph)/2)
def arrow(x1,t1,x2,t2,color=MUT):
    c.setStrokeColor(HexColor(color));c.setFillColor(HexColor(color));c.setLineWidth(1.3)
    c.line(x1,H-t1,x2,H-t2)
    a=math.atan2(t2-t1,x2-x1);q=c.beginPath();q.moveTo(x2,H-t2)
    for da in (-.48,.48):q.lineTo(x2-6*math.cos(a+da),H-(t2-6*math.sin(a+da)))
    q.close();c.drawPath(q,fill=1,stroke=0)
def caption(s):
    global y
    para(s,9,MUT,13)
def table(headers,rows,widths):
    global y
    st=ParagraphStyle('cell',parent=style,fontSize=9,leading=13)
    vals=[[Paragraph(str(z),st) for z in row] for row in [headers]+rows]
    t=Table(vals,colWidths=widths,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),HexColor('#DCECEF')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('ROWBACKGROUNDS',(0,1),(-1,-1),[white,HexColor('#F4F7FA')]),('LINEBELOW',(0,-1),(-1,-1),.5,HexColor('#CBDCE5'))]))
    _,h=t.wrap(R-L,1000);assert y+h<786,(page,y,h);t.drawOn(c,L,H-y-h);y+=h+16
def flow(labels,colors=None,h=58,link=True):
    global y
    n=len(labels);gap=17;w=(R-L-gap*(n-1))/n
    for i,s in enumerate(labels):
        x=L+i*(w+gap);box(x,y,w,h,s,(colors or [TEAL]*n)[i])
        if i<n-1 and link:arrow(x+w,y+h/2,x+w+gap,y+h/2)
        elif i<n-1:text('+',x+w+4,y+h/2-7,12,MUT,'Head')
    y+=h+18

start('Reading guide','From Production to Intelligent Coordination','Historical PI-on edition. For the latest PI-off strategy, see the M4 algorithm booklets and root README.')
para('Imagine a desk-lamp factory. Line A makes bases, B makes heads, and assembly needs both. If orders rise, which line should receive more WIP? If bases are already piling up, will further releases actually help?',13,gap=18)
flow(['A: base processing','A+B: matched pair','Assembly: finished output'])
para('The project starts from these simple-looking but interdependent questions. The algorithm acts as production supervisor, adjusting three WIP caps every 10 minutes to track throughput targets while controlling inventory and adjustment costs.')
note('The lamp factory is a teaching analogy. The code models an abstract A+B assembly system. Diagrams and action examples are explanatory; experiment charts use saved actual results. Illustrative curves are not presented as measured trajectories.')
sub('Suggested reading order')
table(['Topic','Pages','What you will understand'],[
['Problem and environment','2-5','Why more releases are not always better; how simulation runs'],
['Objective and base algorithms','6-7','Reward calculation and original MADQN learning'],
['M4 architecture improvements','8-12','Shared information, memory, coordination, residuals and ensembles'],
['Experiments and results','13-16','Baselines, convergence, performance and ablation evidence'],
['Contribution scope and review','17-18','Supported conclusions and accurate project descriptions']],[105,65,329])
para('Estimated reading time: 25-35 minutes. Each chapter starts with production intuition, explains implementation details and highlights common misunderstandings.',10,MUT)

start('Chapter 1 / Problem','The Factory Needs Matched Output','Research subject: dynamic WIP-cap allocation, not individual machine-speed control.')
para('If A produces many parts and B few, assembly still lacks complete sets. A busy branch may simply build inventory. Plant-wide success depends on finished output meeting the target.')
flow(['A has 3 parts|Buffer full','B has 0 parts|No arrival yet','Assembly waits|Cannot pair A+B'],[GOLD,BLUE,TEAL])
caption('Figure 1. Mismatch illustration: three A parts cannot replace one B part. Not an experiment screenshot.')
sub('Three obvious actions, each with a cost')
table(['Action','Possible outcome','The real question'],[
['Increase all caps','More WIP enters, but bottlenecks may prevent proportional throughput growth','Is WIP insufficient, or is capacity insufficient?'],
['Decrease all caps','Inventory falls, but lines may starve and machines wait','Does inventory saving sacrifice tracking accuracy?'],
['React only to current output error','Repeated increases can cause overshoot, followed by repeated decreases','Has the previous adjustment taken effect yet?']],[105,196,198])
sub('The full research question')
para('In an A+B synchronized assembly system with finite buffers, stochastic processing and changing targets, use observable state to adapt three local CONWIP caps. Track rolling finished-product throughput while reducing WIP and frequent adjustments.')
note('The controlled quantity is permitted WIP circulating in each loop. The controller does not directly change machine speed, machine count or individual job sequencing.')

start('Chapter 2 / Environment','A Small Factory with Three Loops','Environment: PairV2Env. Discrete-event simulation models processing, queues, pairing and blocking. [S1]')
t=y
for row,label,col in [(0,'A',BLUE),(1,'B',GOLD)]:
    top=t+row*76
    for i in range(5):
        x=L+i*59;box(x,top,48,35,f'{label}{i+1}',col)
        if i<4:arrow(x+48,top+17,x+59,top+17)
    box(353,top,73,35,label+'Buffer|Capacity 3',col);arrow(332,top+17,353,top+17)
    arrow(426,top+17,451,t+55)
box(453,t+35,90,43,'Pair A+B',TEAL)
c.setStrokeColor(HexColor(MUT));c.line(498,H-t-78,498,H-t-132);c.line(498,H-t-132,301,H-t-132)
arrow(301,t+132,301,t+154)
for i in range(3):
    x=260+i*96;box(x,t+154,82,46,f"""Assembly {i+1}|2 machines""",TEAL)
    if i<2:arrow(x+82,t+177,x+96,t+177)
text('After pairing, process through assembly stages 1 → 2 → 3',L,t+220,10,MUT)
y=t+247
caption('Figure 2. Actual process topology. Paired material enters assembly stage 1. Each of three assembly stages has two parallel machines.')
table(['Section','Stages and machines','Baseline mean processing time (minutes)'],[
['A branch','5 stages, 1 machine each','7, 8, 7, 8, 7'],['B branch','5 stages, 1 machine each','8, 7, 8, 7, 7'],['Assembly line','3 stages, 2 machines each','12, 16, 12']],[92,197,210])
para('Stages have FIFO queues. A/B parts enter their finished-parts buffers after processing. Full buffers can block branch-end machines from releasing parts. Assembly admits a pair only when both parts and assembly-cap allowance are available.')
note('Nominal bottleneck capacity is approximately 7.5 parts/hour under baseline conditions. Actual output also depends on WIP caps, randomness, pairing waits and finite buffers. Nominal capacity is not actual throughput.')

start('Chapter 2 / Control actions','CONWIP: Issue Production Passes','Separate local caps manage A, B and assembly, rather than one plant-wide shared token pool. [S1]')
para('Think of cap as a number of passes. cap_A=3 permits up to three parts to occupy A-loop slots. A part leaving releases a slot, allowing replacement work. A/B buffers have separate capacity limits.')
flow(['Observe|Caps (3,3,2)','Choose action|(+1,0,-1)','New caps|(4,3,1)'])
caption('Figure 3. Cap update illustration. Code indices 0/1/2 mean decrease by 1, hold, increase by 1; indices are not increments.')
sub('Reducing a cap does not delete parts')
para('If assembly already has 2 sets in process and its cap falls from 2 to 1, both sets continue. New releases pause until WIP falls naturally. Cap constrains releases; instantaneous WIP may temporarily exceed a reduced cap.')
table(['Parameter','Current value','Intuitive meaning'],[
['Initial caps','(3,3,2)','Provide basic flow through all three units'],['Cap bounds','0-8 per line','Limit releases or pause replenishment'],['Decision interval','10 minutes','Let the factory run between decisions'],['Joint action count','27','Three choices per line: 3×3×3'],['Control horizon','3600 minutes / 360 steps','Evaluate 60 hours of control per episode'],['Warm-up','720 minutes','Run 12 hours before evaluating control']],[100,153,246])
note('With demand_gated=False, releases are mainly cap-constrained, without an additional no-order/no-release gate. Targets affect control through reward; overproduction also incurs tracking cost.')

start('Chapter 3 / Challenges','Why This Is Not Simple Addition and Subtraction','Pairing constraints, delays, randomness and regime changes interact.')
flow(['Increase A cap','Multiple A stages','Wait for matching B','Output after assembly'],h=48)
caption('Figure 4. Delay chain. Actions affect releases first, then finished output after processing and pairing. Arrows do not imply fixed durations.')
para('This separates action from reward. A present shortage may reflect insufficient releases several steps ago or simply slow processing. Repeatedly increasing caps before earlier effects arrive can cause overreaction.')
sub('Six regimes test the controller')
table(['Control time (minutes)','Target (parts/hour)','Change'],[
['0-600','3','Baseline processing'],['600-1200','6','Higher target'],['1200-1800','5','A/B Gamma shape: 9 → 2'],['1800-2400','5','Variability restored; B stage 3 mean +15%'],['2400-3000','5','B restored; A stage 2 mean +15%'],['3000-3600','3','Lower target; processing means restored']],[127,110,262])
para('Smaller Gamma shape means more variability at the same mean. A/B coefficients of variation rise from approximately 0.33 to 0.71. Mean speed does not directly slow, but completion timing becomes less predictable and pairing waits may increase.')
note('Change times are predefined. Test seeds vary random processing samples, not the entire regime schedule. Results show adaptation to randomness within this scenario, not effectiveness for every unknown demand curve.')

start('Chapter 4 / Objective',"Reward: Calculate the Plant's Total Cost",'Tracking, inventory and adjustments determine one shared team reward. [S1]')
flow(['Tracking error|Weight 1','WIP cost|Weight 0.05','Cap-change cost|Weight 0.005'],[BLUE,TEAL,GOLD],link=False)
para("Each step's reward is the negative sum of these costs. Tracking uses squared relative throughput error. Inventory uses mean part-equivalent WIP over the 10-minute interval divided by 38. Action cost uses the mean absolute cap change across three units.")
note('r = -[(throughput - window target) / max(window target,1)]² - 0.05 × interval mean WIP / 38 - 0.005 × Σ|cap change| / 3. Scheduled changes align with decision boundaries; code also handles within-interval regime splits.')
sub('A hand-calculation example (illustrative)')
para('Suppose rolling throughput is 4.5 parts/hour, the window target 5, mean WIP 15, and one line increases its cap by 1. Tracking cost is 0.01, inventory cost approximately 0.01974, adjustment cost approximately 0.00167. Reward is approximately -0.03140, or -31.40 after multiplying by 1000.')
para('This explains the trade-off between smaller error and lower inventory. Weights express operating preferences. These results use the fixed stated weights and do not automatically transfer to a different cost structure.')
sub('A two-hour window reduces noise but adds lag')
para('Throughput uses completions over the last 120 minutes, and the reference averages target history over the same window. When the target jumps from 3 to 6, the scoring reference transitions gradually rather than revaluing the entire past window at 6. Observations include both current and window targets.')
table(['Metric','Correct interpretation'],[['reward×1000','Mean per-step episode reward ×1000, aggregated across episodes; higher is better'],['NRMSE','Square root of the episode mean squared tracking error; lower is better'],['Mean WIP','One part per A/B WIP; two per assembly set; plus buffer parts'],['Cap-change points',"Decisions where any cap changes, not the sum of three lines' adjustment magnitudes"]],[100,399])

start('Chapter 5 / Starting point','Original MADQN: Three Learning Supervisors','One local Q network per line, sharing team reward but separately estimating action values. [S2]')
flow(['45-D current observation','Line-specific Q network','Decrease / hold / increase|Three action scores'])
caption('Figure 5. One-line original-MADQN decision structure. Each of the three lines has such a network.')
para('Q estimates the expected future return from taking an action now. Training gathers experience through repeated simulation. Execution selects the highest-valued legal action. Q is neither predicted throughput nor a success probability.')
sub('How does one learning update happen?')
flow(['Current state s','Take action a','Receive reward r|Observe next state s′','Update action values'],h=54)
para('The original implementation stores state, action, reward and next state in replay, then samples experiences randomly. Double-DQN-style targets use the online network for next-action selection and the target network for evaluation, reducing some overestimation.')
sub('What does it know, and what is missing?')
para("Local agents are not entirely blind to the plant. Their 45-D observations include own WIP/cap, shared buffers, rolling throughput, targets, recent busy/idle/blocked summaries and stage states. However, they lack complete broadcasts of all units' WIP/operating summaries and explicit observation-sequence memory.")
para("During independent learning, teammates' policies also change. Outcomes depend on all three agents, not only one. Shared reward reports whether the team did well but does not identify which decision helped.")
note('This motivates M4: add cross-line and temporal information, strengthen team-value learning, and exploit PI knowledge and independent trained models.')

start('Chapter 6 / M4 overview','Connect Production Information, Experience and Learning','This historical PI-on M4 retains QMIX. Independent Q ensembling does not remove centralized training. [S3, S4]')
t=y
flow(['Local observation: 45-D','Shared summary: 24-D','PI action encoding: 3-D'],h=44,link=False)
box(139,y,317,44,'Per agent: 72-D input × latest 4 frames');arrow(297,y-14,297,y);y+=62
box(139,y,317,44,'GRU + three line-specific Q heads');arrow(297,y-18,297,y);y+=65
box(48,y,229,52,'Training: QMIX team value|3-step + PER',BLUE)
box(307,y,240,52,'Execution: five-model Q average|Select residual and map to action',TEAL)
arrow(240,y-22,165,y);arrow(355,y-22,420,y);y+=72
caption('Figure 6. Historical M4 architecture. Dimensions, history and modules follow implementation. Training and execution branches have different roles.')
para('Within a member, three agents share a GRU encoder but have line-specific heads. Five members train separately with different seeds. Execution still broadcasts global summaries. Decentralized action selection does not mean communication-free execution.')
table(['Version','Added to the preceding version','Problem addressed'],[['M1','Global features + GRU','Incomplete information and unclear trends'],['M2','QMIX-style team-value mixing','Cross-line action coordination'],['M3','3-step return + PER','Delayed production feedback and learning efficiency'],['M4','Residual actions guided by current PI','Difficult exploration from scratch; reusable control knowledge'],['Final historical design','Longer training + five-member independent Q ensemble','Bias and instability from individual training runs']],[72,220,207])

start('Chapter 6 / Information and memory','Global Dashboard + GRU: Understand Change','Mechanisms motivate the design; ablations must establish actual benefits.')
para("An A supervisor sees low throughput and plans a cap increase. The shared dashboard shows A's buffer full, B undersupplied and assembly waiting for B. Extra A releases may not help. Coordination information enables decisions beyond local interests.")
flow(['A WIP / cap','B WIP / cap','Assembly WIP / cap'],h=45,link=False)
box(L,y,R-L,45,'Broadcast summary: buffers, throughput/targets and busy/idle/blocked/pair-starvation states');y+=63
caption('Figure 7. The 24-D global summary uses observable information, not future processing times or random-number access.')
sub('Why can the same snapshot require different actions?')
table(['Current throughput is below target in every case','Possible history','Control implication'],[['Case A','Persistent shortage with no recovery','Supply may need sustained correction'],['Case B','Previously lower, now recovering','The previous adjustment may be taking effect'],['Case C','A single short-lived fluctuation','A large correction may overreact']],[123,193,183])
flow(['Earlier observation','Previous observation','Current observation','GRU summary|Pass to Q head'],h=48)
para('The network re-encodes the latest 4 frames on every call. They span three 10-minute intervals, not unlimited episode-wide memory. GRU learns which changes to retain but is not guaranteed to identify every true cause.')
note('Original MADQN already sees recent operating summaries. M1 adds more complete cross-line broadcasts and explicit sequence processing. Do not describe the original as completely blind.')

start('Chapter 6 / Team learning','What QMIX, n-Step and PER Each Do','They change team-value representation, learning targets and replay sampling, respectively.')
flow(['A action value','B action value','Assembly action value'],h=43,link=False)
box(144,y,310,46,'QMIX: combine with global state into team Q',BLUE);y+=63
caption('Figure 8. QMIX mixes line values into team value during training; it does not divide reward equally among agents.')
para('QMIX constrains the mixer to be monotonic, aligning local and team value directions so agents can select actions separately. It cannot express every possible joint-action relationship or produce a fully independent true-contribution reward for each supervisor.')
sub('3-step: bring delayed feedback into targets sooner')
note('Target illustration: current reward + γ×next reward + γ²×following reward + γ³×successor-value estimate. γ=0.99. Termination removes subsequent bootstrapping.')
para('Compared with one-step reward, three-step targets directly bring more actual feedback into an update. Each step is 10 minutes, but the algorithm does not look only 30 minutes ahead: the bootstrapped value represents a longer future.')
sub('PER: revisit experiences with inaccurate estimates')
flow(['Store in replay','Compute TD error','Raise sampling probability','Update network|Correct estimates'],h=46)
para('TD error is the gap between the current estimate and learning target. PER favors larger gaps and uses importance weights to mitigate sampling bias. Like a mistake notebook, it may also focus on noisy cases; it does not automatically identify the most important bottlenecks.')
note("In the experiments, M2's mean reward worsens relative to M1, then M3 improves it. Evaluate designs with data rather than module count.")

start('Chapter 6 / Experience guidance','PI Suggests; M4 Learns Corrections','The PI baseline uses fixed controller parameters. Historical files label it as a reconstruction. [S2, S3]')
para('PI reads rolling throughput error. Its proportional term responds to current deviation and its integral term accumulates sustained deviation. The implementation also estimates workload allocation from processing means, adjusts target caps using buffer pressure, and uses filtering, deadbands and integral limits. It is not simply increase all caps whenever output is low.')
flow(['PI proposed action','M4 residual|-1 / 0 / +1','Legality check','Actual cap increment'],h=52)
caption('Figure 9. Residuals act on discrete actions. Zero follows PI; it is not an arbitrary real-valued cap addition.')
table(['PI proposal','Feasible residual','Possible executed action'],[['Decrease cap','0 or +1','Decrease or hold'],['Hold','-1, 0 or +1','Decrease, hold, increase'],['Increase cap','-1 or 0','Hold or increase']],[106,185,208])
para('If PI proposes increasing A, M4 can change it to hold but not directly to decrease, because the residual spans only adjacent actions. At cap=0 or cap=8, remove boundary-violating choices. This reduces exploration space but also limits corrections.')
sub('Why is this suitable for the environment?')
para('PI already provides basic target/state-based cap control. M4 can focus on pairing imbalance, variability, delayed responses and coordination across regimes. Early training also uses PI guidance and a zero-residual preference.')
note('PI guidance is not a formal safety proof and does not guarantee every correction beats PI. This historical M4 is a hybrid controller with a PI prior, not unaided pure MADQN.')

start('Chapter 6 / Model ensemble','Five Sets of Scores, One Action','The historical final version averages Q values from five independent members, then selects per-line actions. [S4]')
para("Different initialization and experiences produce different Q estimates even with the same algorithm. Ensembling aims to average away some individual bias. All members see the same state and PI proposal, then score each line's residual actions.")
table(['Illustrative model','Residual -1','Residual 0','Residual +1'],[['Member A','0.60','0.58','0.10'],['Member B','0.45','0.59','0.20'],['Member C','0.48','0.57','0.18'],['Mean Q','0.51','0.58','0.16']],[130,123,123,123])
caption('Figure 10. Teaching example with three members. Average scores first, then select residual 0. Actual deployment uses five; these are not experimental Q values.')
para("This is not majority voting: each member's score margin matters. Nor are output caps averaged and rounded. Execution produces one definite legal discrete action.")
sub('Independent Q versus joint Q')
para("The selected independent Q ensemble averages local Q values per A/B/assembly unit and takes separate argmax choices. QMIX remains in each member's training. The earlier joint-Q experiment instead used mixers to evaluate joint actions. Their execution rules differ.")
table(['Same plateau-selected members: test results','reward×1000','NRMSE'],[['Mean when 5 members execute separately','-38.6484','0.13233'],['Execution as five-member Q ensemble','-36.8359','0.12642']],[260,120,119])
para('This more directly supports an ensemble benefit: the models are identical, execution differs. However, the rows summarize 60 member-seed episodes versus 12 ensemble episodes, not equal numbers of independent training experiments.',10)
note('Inference runs five members. Highly correlated errors may limit gains, and member Q scales affect averaging. Improvement is not universally guaranteed.')

start('Chapter 7 / Baselines','Each Baseline Answers a Different Question','Baselines explain where improvement comes from, rather than merely increasing the comparison count.')
table(['Method','Supervisor analogy','Comparison purpose'],[['Fixed CONWIP','Choose caps beforehand and hold them','Assess dynamic adjustment versus fixed settings'],['Current coordinated PI','Respond to error, accumulated deviation and material pressure','Compare with interpretable feedback control'],['Original MADQN','Three learners separately estimate local action values','Base reference for the improvement chain'],['Centralized DQN','One supervisor reads 135-D state and selects 27 joint actions','Compare centralized decisions and multi-agent structure'],['Learned-model MPC + PI','Predict, search future actions and constrain deviation using PI','Assess this predictive-control implementation']],[113,193,193])
sub('The simplest distinction between MPC and Q learning')
flow(['Q learning|Direct action valuation','MPC|Predict candidate futures'],[TEAL,BLUE],h=57)
para('Q networks compress long-term experience into action values and mainly evaluate them at execution. MPC predicts what an action will cause, searches action sequences, executes only the first action and replans. Their learned objects differ.')
para('This MPC trains three feedforward predictors on 17280 transitions to predict observation increments and immediate reward. It uses a 4-step horizon and beam width 8, later adding a PI anchor, deviation penalties and a predicted-cost fallback. It does not implement the previously proposed GRU predictor.')
note('Initial MPC performance was poor, and its controller was subsequently revised after inspecting test/holdout performance. Results are exploratory; neither set remains a fully unseen final test for MPC. PI safety region names an action constraint, not a formal safety guarantee. [S6]')

start('Chapter 8 / Training','When Has a Learning Curve Reached a Plateau?','Training, validation and test rewards serve different purposes. [S4, S5]')
flow(['Training|Explore and update','Validation|Select checkpoint','Test|Evaluate frozen policy'],h=54)
para('Training reward varies with exploration and a changing policy. Validation disables exploration on fixed seeds, helping assess improvement. Test reports selected-model results. Repeated test-driven algorithm changes require a new final test set.')
sub('Actual validation curves: five members')
top=y;xx=L+39;yy=top+22;ww=R-xx-12;hh=146
colors=[TEAL,BLUE,GOLD,'#885DAB','#D05C6A']
for v in [-450,-350,-250,-150,-50]:
    py=yy+hh-(v+450)/420*hh;c.setStrokeColor(HexColor('#E2EAF0'));c.line(xx,H-py,xx+ww,H-py);text(str(v),L,py-5,8,MUT)
for k,seed in enumerate(range(13101,13106)):
    rows=list(csv.DictReader((ROOT/f'outputs/madqn_m4_final_converged/run_{seed}/validation.csv').open(encoding='utf-8-sig')))
    vals=[(float(r['episode']),float(r['validation_reward_x1000'])) for r in rows]
    assert all(-450<=v<=-30 for _,v in vals)
    p=c.beginPath()
    for j,(ep,v) in enumerate(vals):
        px=xx+ep/1250*ww;py=H-(yy+hh-(v+450)/420*hh)
        if j==0:p.moveTo(px,py)
        else:p.lineTo(px,py)
    c.setStrokeColor(HexColor(colors[k]));c.setLineWidth(1.3);c.drawPath(p)
    text(str(seed),L+k*97,top+190,9,colors[k],'Head')
for ep in [0,250,500,750,1000,1250]:text(str(ep),xx+ep/1250*ww-8,yy+hh+5,8,MUT)
text('Validation reward ×1000 (higher is better)',L,top,9,MUT);y=top+217
caption('Figure 11. Unsmoothed validation reward against training episode. All observations remain, including early sharp drops.')
para('Stopping: at least 300 episodes, validation every 50, and stop after 4 checks without sufficient improvement. Main threshold: reward×1000 gain of 0.05, with an NRMSE tie-break. Five members stopped at 900, 750, 750, 1250 and 750 episodes. Use their best validation checkpoints.',10)
note('Convergence here means a practical validation-plateau stopping rule, not mathematical convergence, no possible future improvement or global optimality.')

start('Chapter 8 / Results','Where Does Final Historical M4 Improve?','Existing test summary: higher reward and lower NRMSE are better. [S5, S6]')
table(['Method','reward×1000','NRMSE','Mean WIP','Change points'],[['Fixed CONWIP','-116.570','0.31025','15.370','1.00'],['Current PI','-47.978','0.16769','14.452','129.58'],['Original MADQN','-65.659','0.19686','16.851','351.00'],['Central DQN','-62.991','0.17949','20.086','351.82'],['MPC + PI','-54.038','0.18339','14.626','157.42'],['Final historical M4 ensemble','-36.836','0.12642','14.692','231.92']],[127,108,87,90,87])
top=y
for i,(name,value,col) in enumerate([('Fixed',.31025,MUT),('PI',.16769,BLUE),('MADQN',.19686,MUT),('DQN',.17949,MUT),('MPC+PI',.18339,MUT),('M4 ensemble',.12642,TEAL)]):
    t=top+i*25;text(name,L,t,10,INK);x=L+74;bw=value/.35*345
    c.setFillColor(HexColor(col));c.roundRect(x,H-t-16,bw,14,3,fill=1,stroke=0);text(f'{value:.5f}',x+bw+7,t,9,MUT)
y=top+161;caption('Figure 12. Actual mean NRMSE, bars starting at zero. Shorter is better. No error bars or significance claim.')
para('Versus original MADQN, final historical M4 reduces NRMSE approximately 35.8%, aggregate cost 43.9%, mean WIP 12.8%, and change points 33.9%. These descriptive whole-design differences include longer training and ensemble benefits.',10)
para('Versus PI, M4 reduces NRMSE approximately 24.6% and aggregate cost 23.2%, but WIP is approximately 1.7% higher and change points 79% higher. Its advantage is tracking, not uniformly lower operating metrics.',10)
note("Original MADQN/DQN summarize 5 models × 12 seeds. Final ensembling uses one five-member combination on 12 seeds. Fixed policy's 1 change point follows existing statistics, not continuous dynamic adjustment. Rows are not strict equal-budget, equal-independent-run comparisons.")

start('Chapter 8 / Evidence','Interpret Module Effects at the Right Level','Ablations examine module additions; final-system comparisons examine the entire package.')
table(['300-episode stage','reward×1000','NRMSE','Relative to previous version'],[['M0 original MADQN','-65.659','0.197','Existing baseline'],['M1 global features + GRU','-50.098','0.153','Information/sequence version improves'],['M2 adds QMIX','-51.644','0.163','Tracking and reward do not improve further'],['M3 adds n-step + PER','-49.720','0.155','Improves over M2'],['M4 adds PI residuals','-43.066','0.145','Best within this version group']],[176,99,75,149])
para('This table does not prove QMIX useless or every component useful. Components can interact. Removing QMIX from final M4 requires direct matched-budget, matched-seed comparison. M0 also differs in training implementation, so M0→M1 is not a clean single-variable ablation. [S3]')
sub('Final ensemble on two seed sets')
table(['Evaluation set','reward×1000 (mean ± SD)','NRMSE (mean ± SD)'],[['test: 12 seeds','-36.8359 ± 1.7177','0.12642 ± 0.00677'],['holdout: 20 seeds','-36.3854 ± 2.1638','0.12451 ± 0.00858']],[117,191,191])
para('Similar means support some stability across random processing samples within this scenario. SD measures episode variation, not confidence intervals. Samples are small and there is only one ensemble combination; newly trained ensembles need not perform identically.')
sub('A more defensible conclusion')
para('Existing records show final historical M4 improves tracking and aggregate reward over the listed PI and learning baselines, and reduces inventory and changes versus original MADQN. NRMSE&lt;0.10 remains unmet. Not every improvement has separately demonstrated positive benefit.')
note('For confirmatory publication results, freeze algorithms/hyperparameters, use new seeds and regime schedules, match budgets and analyze paired outcomes. Current data illustrate motivation and observed effects.')

start('Chapter 9 / Contributions','Place Contributions Where the Evidence Supports Them','A combined design and empirical validation for a specific production-control problem.')
para('GRU, QMIX, n-step, PER, residual control and ensembles are established techniques. The contribution is aligning them with distinct A+B assembly challenges and organizing a working controller under a shared production objective. This does not establish literature-level priority.')
table(['Environment challenge','Corresponding design','Defensible value'],[['Upstream/downstream pairing and finite buffers','Global summaries + team-value training','Help each line learn from plant-wide information'],['Processing delays and timing variability','GRU + multi-step returns','Use recent dynamics and more distant feedback'],['Inefficient exploration from scratch','PI action residuals and training guidance','Learn corrections on top of feedback control'],['Single-model training differences','Five-member independent Q averaging','This member combination executes better than the individual mean']],[141,171,187])
sub('Research summary')
para('This project studies dynamic CONWIP-cap control in stochastic paired assembly. A three-loop simulation with changing operating regimes captures finite buffers, material synchronization and delayed feedback. The historical MADQN-based design integrates global features, short-history GRU, QMIX team training, multi-step prioritized replay and PI residual guidance. Averaging local Q values across five independently trained members improves tracking and aggregate reward in existing scenarios, with explicit discussion of inventory, adjustments and evaluation fairness.')
sub('Which benefits remain unverified?')
para('Real-factory deployment, breakdown robustness, transfer to new topologies, arbitrary target schedules and large-scale multi-line expansion remain untested. Cap actions do not replace maintenance, equipment configuration or order scheduling.')
note('For mechanisms, say aims to mitigate or may help. For results, say observed in this experiment. Plausible design and experimental proof are different levels of evidence.')

start('Appendix / Review and evidence','Review the Project with These Six Questions','The guide describes the implemented system and its recorded experiments.')
table(['Question','Answer'],[['Are cap and WIP identical?','No. Cap is an allowance; WIP is actual inventory. WIP may temporarily exceed a reduced cap.'],['Why not set every cap to 8?','More inventory may not remove a bottleneck, and raises inventory cost and overproduction risk.'],['Does M4 require PI?','This historical version does: PI supplies inputs and an action anchor. The latest PI-off version does not.'],['Does independent Q ensembling remove QMIX?','No. Members train with QMIX; execution averages Q per line.'],['Does convergence imply optimality?','No. It means a practical validation-plateau stopping rule here.'],['Does M4 outperform PI on everything?','Tracking and reward improve; PI has lower inventory and fewer change points.']],[190,309])
sub('Local source index')
para('[S1] work/conwip_four_strategy_v3_madqn/conwip/pair_v2.py: topology, parameters, actions, observations and reward.<br/>[S2] baseline_algorithms.py in the same directory: current PI, original MADQN and central DQN.<br/>[S3] work/conwip_four_strategy_v3_madqn/run_madqn_ablation.py; outputs/madqn_ablation/MADQN_Ablation_Report.md: M1-M4 definitions and early results.<br/>[S4] work/train_m4_until_converged.py; work/evaluate_m4_converged_final.py: stopping, member selection and Q ensembling.<br/>[S5] outputs/madqn_m4_final_converged/: training_manifest.json, comparison_final.json, validation.csv and comparison tables.<br/>[S6] work/train_mpc_controller.py; outputs/mpc_model_predictive/comparison_mpc.json: MPC implementation and exploratory results.',9,gap=12)
para('Paths are relative to the project root. The PI baseline is reconstructed from the available controller parameters. Experiment values are reported without modification.',9,MUT)
note('You can now explain why a busy A line may not increase finished output, why current error alone is insufficient, why M4 uses shared information and memory, and why better results do not attribute all gains to one module.')
c.save()
reader=PdfReader(str(P));assert len(reader.pages)==18
for i,p in enumerate(reader.pages):
    assert len(p.extract_text())>150,(i,'empty')
print(P);print('pages',len(reader.pages))
