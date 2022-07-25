import os
import pathlib

from PIL import Image, ImageFont, ImageDraw

import helper as hp


def save_image(code_txt, img_file):
    code_l, code_w = hp.get_method_stats(None, code_txt)
    img_h, img_w = (code_l + 1) * 70, (code_w + 1) * 23
    img = Image.new(mode=hp.IMG_MODE, size=(img_w, img_h), color="#FFFFFF")
    draw = ImageDraw.Draw(img)
    fnt = ImageFont.truetype(font=hp.DEF_FONT_PATH, size=50)
    draw.text(xy=(25, 50), text=code_txt, spacing=50, font=fnt, fill="black")
    if not pathlib.Path(img_file).is_file():
        pathlib.Path(os.path.dirname(img_file)).mkdir(parents=True, exist_ok=True)
    img.save(img_file, dpi=(600, 600))
    # print("Saved {}".format(img_file))


def main():
    for db in hp.DATABASE:
        for pt in hp.PARTITIONS:
            ex = 0
            for file in pathlib.Path(hp.SRC_PATH + "{}/{}".format(db, pt)).glob('**/*.java'):
                if not os.path.isdir(file) and str(file).endswith(hp.SRC_EXT):
                    # print("Reading {}".format(file))
                    code_txt = pathlib.Path(file).read_text().strip()
                    method_name = str(file).split('_')[-1].replace(hp.SRC_EXT, '')
                    if code_txt.count(" {}(".format(method_name)) > 0:
                        code_txt = code_txt.replace(" {}(".format(method_name), " {}(".format("METHOD_NAME"), 1)
                        img_file = str(file).replace(hp.SRC_PATH, hp.IMG_PATH.format("original_methods"))\
                            .replace(hp.SRC_EXT, hp.IMG_EXT)
                        save_image(code_txt, img_file)
                        ex += 1
            print("Done: {}-{}: #{}".format(db, pt, ex))
            print("\n")


if __name__ == "__main__":
    main()
