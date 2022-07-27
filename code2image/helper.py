import pathlib
import subprocess

ROOT_PATH = "/scratch/deployment/code-image/"
SRC_PATH = "/scratch/data/java_method_raw/"
IMG_PATH = "/scratch/data/java_method_img/{}/"
DEF_FONT_PATH = ROOT_PATH + "/others/default.ttf"

JAR_JAVA_METHOD = ROOT_PATH + "/JavaLoader/target/jar/JavaLoader.jar"

METHOD_MASK = "METHOD_NAME"

SRC_EXT = ".java"
IMG_EXT = ".png"
IMG_MODE = "L"

MAX_WIDTH = 120
MAX_LENGTH = 30

DATABASE = ["java-top10", "java-top50"]
PARTITIONS = ["training", "validation", "test"]


def get_method_stats(src_fie, code_txt):
    if not code_txt or len(code_txt.strip()) == 0:
        code_txt = pathlib.Path(src_fie).read_text().strip()
    code_txt = code_txt.split('\n')
    max_l, max_w = len(code_txt), len(max(code_txt, key=len))
    return [max_l, max_w]


def get_method_body(src_file):
    cmd = ['java', '-jar', JAR_JAVA_METHOD, src_file]
    contents = subprocess.check_output(cmd, encoding="utf-8", close_fds=True)
    return str(contents).strip()
