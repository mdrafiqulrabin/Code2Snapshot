import pathlib

ROOT_PATH = "/scratch/deployment/code-image/"
SRC_PATH = "/scratch/data/java_method_raw/"
IMG_PATH = "/scratch/data/java_method_img/{}/"
DEF_FONT_PATH = ROOT_PATH + "/others/default.ttf"

METHOD_MASK = "METHOD_NAME"

SRC_EXT = ".java"
IMG_EXT = ".png"
IMG_MODE = "L"

DATABASE = ["java-top10", "java-top50"]
PARTITIONS = ["training", "validation", "test"]


def get_method_stats(src_fie, code_txt):
    if not code_txt or len(code_txt.strip()) == 0:
        code_txt = pathlib.Path(src_fie).read_text().strip()
    code_txt = code_txt.split('\n')
    max_l, max_w = len(code_txt), len(max(code_txt, key=len))
    return [max_l, max_w]
