"""Create user-facing material packages and a comparison sheet for the three examples."""
from pathlib import Path
import shutil
from PIL import Image,ImageDraw,ImageFont
from lens_material import ROOT,prepare_material

def main():
    names=[('下载美瞳超清图片.png','01_灰棕放射纹'),
           ('生成美瞳超清晰图片 (1).png','02_粉棕花瓣纹'),
           ('生成美瞳超清晰图片 (2).png','03_深棕渐变纹')]
    target=ROOT/'处理后美瞳';target.mkdir(exist_ok=True)
    sheet=Image.new('RGB',(1200,990),(235,235,235))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',19)
    draw=ImageDraw.Draw(sheet)
    for i,(filename,label) in enumerate(names):
        material=prepare_material(ROOT/filename)
        folder=target/label;folder.mkdir(exist_ok=True)
        for resource in ['lens_rgba.png','material.json','calibration.png','preview_dark.png','preview_light.png']:
            shutil.copy2(material.directory/resource,folder/resource)
        draw.text((i*400+12,10),label,font=font,fill='black')
        for j,(resource,text) in enumerate([('calibration.png','原图与内外边界'),('preview_dark.png','深色背景'),('preview_light.png','浅色背景')]):
            image=Image.open(folder/resource);image.thumbnail((300,270))
            sheet.paste(image,(i*400+50,j*310+65))
            draw.text((i*400+12,j*310+38),text,font=font,fill='black')
    sheet.save(target/'三款处理对比.png')
    (target/'说明.txt').write_text('原始三张图片保持不变。各目录 lens_rgba.png 是可直接导入的透明花纹。\n'
        'material.json 保存原图来源、边界与反光修补参数。商品图的透明度和被反光遮住的细节是近似重建，尤其第三款仍需实物对照。\n'
        '软件直接导入原图也会自动使用对应处理结果；可点击“校准素材”调整中心、内外圈和反光修补。\n',encoding='utf-8')
    print('Processed examples:',target)

if __name__=='__main__':main()
