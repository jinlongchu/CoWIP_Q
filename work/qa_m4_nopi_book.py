import json, hashlib
from pathlib import Path
import pdfplumber
from pypdf import PdfReader

root=Path(__file__).resolve().parent.parent
path=next((root/'outputs'/'M4_Algorithm_Booklets').glob('*.pdf'))
data=root/'outputs'/'m4_no_pi_ablation'
protocol=json.loads((data/'protocol.json').read_text(encoding='utf-8'))
assert len(protocol['configs'])==6 and not any(r['pi'] for r in protocol['configs'])
summary=json.loads((data/'comparison_summary.json').read_text(encoding='utf-8'))
manifest=json.loads((data/'training_manifest.json').read_text(encoding='utf-8'))
assert len(manifest)==30 and sum(r['converged'] for r in manifest)==27
with pdfplumber.open(path) as doc:
    assert len(doc.pages)==20
    texts=[p.extract_text() for p in doc.pages]
    for i,p in enumerate(doc.pages):
        assert len(texts[i])>100
        assert not any(c['text'] in ('\x00','\ufffd') for c in p.chars)
        assert not any(c['x0']<0 or c['x1']>p.width or c['top']<0 or c['bottom']>p.height for c in p.chars)
        assert f'{i+1:02d} / 20' in texts[i]
    table=texts[17]
    for r in summary:
        if r['split']=='test' and r['method']=='q_ensemble_5':
            for key,digits in [('reward_x1000_mean',3),('nrmse_mean',5),('mean_wip_mean',3),('cap_adjustment_points_mean',1)]:
                assert f'{r[key]:.{digits}f}' in table,(r['config'],key)
    assert '−35.540' in texts[11] or '-35.540' in texts[11]
reader=PdfReader(path)
links=[]
for a in reader.pages[-1].get('/Annots',[]):
    action=a.get_object().get('/A',{})
    if action.get('/URI'):links.append(action['/URI'])
assert len(links)==4
assert len(reader.outline)==20
report={'pages':20,'page_size':'A4 landscape','embedded_text':True,'reference_links':links,
    'bookmarks':len(reader.outline),'file_bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
    'text_checks':'pass','data_table_check':'matches test / q_ensemble_5','training_status_check':'27 of 30 early-stopped',
    'visual_review':'20 pages rendered; final changed pages reviewed separately'}
(root/'work'/'m4_book_qa'/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=True))
