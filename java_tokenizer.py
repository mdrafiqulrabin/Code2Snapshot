import subprocess
from pathlib import Path
import json

JAR_JAVA_TOKENIZER = "JavaTokenizer/target/jar/JavaTokenizer.jar"


def get_tokens(file_path):
    cmd = ['java', '-jar', JAR_JAVA_TOKENIZER, file_path]
    contents = subprocess.check_output(cmd, encoding="utf-8", close_fds=True)
    return str(contents).strip()


if __name__ == '__main__':
    input_path = 'sample_input.java'
    program1 = Path(input_path).read_text()
    print("\noriginal program:\n{}".format(program1))
    program2 = get_tokens(input_path)
    print("\ntokenized program:\n{}".format(json.loads(program2)["tokens"]))
