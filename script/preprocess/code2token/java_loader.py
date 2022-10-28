import subprocess
from pathlib import Path

JAR_JAVA_LOADER = "JavaLoader/target/jar/JavaLoader.jar"


def load_method(file_path):
    cmd = ['java', '-jar', JAR_JAVA_LOADER, file_path]
    contents = subprocess.check_output(cmd, encoding="utf-8", close_fds=True)
    return str(contents).strip()


if __name__ == '__main__':
    input_path = 'sample_input.java'
    program1 = Path(input_path).read_text()
    print("\noriginal program:\n{}".format(program1))
    program2 = load_method(input_path)
    print("\nrefactored program:\n{}".format(program2))
