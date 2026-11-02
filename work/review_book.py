from pathlib import Path
from PIL import Image,ImageOps,ImageDraw
p=Path(__file__).parent/'tmp/pdfs'
pages=sorted(p.glob('page-*.png'))
assert len(pages)==18
for start in range(0,18,6):
    sheet=Image.new('RGB',(1200,1740),'#cbd5dd')
    for j,f in enumerate(pages[start:start+6]):
        im=Image.open(f).convert('RGB');im.thumbnail((580,820))
        x=(j%2)*600+10;y=(j//2)*580+20
        im.thumbnail((390,550))
        # Three columns and two rows preserve readable full-page previews.
        x=(j%3)*400+5;y=(j//3)*870+28
        im=Image.open(f).convert('RGB');im.thumbnail((390,825))
        sheet.paste(im,(x,y));ImageDraw.Draw(sheet).text((x,y-18),f.name,fill='black')
    sheet.save(p/f'contact-{start//6+1}.png')
