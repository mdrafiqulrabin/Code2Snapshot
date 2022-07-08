import subprocess

ROOT_PATH = "/scratch/deployment/code-image/"
SRC_PATH = "/scratch/data/java_method_raw/"
TOKEN_PATH = "/scratch/data/java_method_token/"
JAR_JAVA_TOKENIZER = ROOT_PATH + "/JavaTokenizer/target/jar/JavaTokenizer.jar"

METHOD_MASK = "METHOD_NAME"

SRC_EXT = ".java"
TOKEN_EXT = ".json"

DATABASE = ["java-top10", "java-top50"]
PARTITIONS = ["training", "validation", "test"]


def get_method_token(src_file):
    cmd = ['java', '-jar', JAR_JAVA_TOKENIZER, src_file]
    contents = subprocess.check_output(cmd, encoding="utf-8", close_fds=True)
    return str(contents).strip()
