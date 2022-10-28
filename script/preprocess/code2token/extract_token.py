import os
import pathlib

import helper as hp


def main():
    for db in hp.DATABASE:
        for pt in hp.PARTITIONS:
            json_file = hp.TOKEN_PATH + "{}/{}".format(db, pt) + hp.TOKEN_EXT
            if not pathlib.Path(json_file).is_file():
                pathlib.Path(os.path.dirname(json_file)).mkdir(parents=True, exist_ok=True)
            with open(json_file, 'w') as f_json:
                ex = 0
                for file in pathlib.Path(hp.SRC_PATH + "{}/{}".format(db, pt)).glob('**/*.java'):
                    if not os.path.isdir(file) and str(file).endswith(hp.SRC_EXT):
                        # print("Reading {}".format(file))
                        json_entry = hp.get_method_token(file)
                        f_json.write(json_entry + "\n")
                        ex += 1
                print("Done: {}-{}: #{}".format(db, pt, ex))
                print("\n")


if __name__ == "__main__":
    main()
