ROOT_PATH = "/scratch/data/java_method_img/"

IMG_TYPES = {
    "original": "original_methods",
    "reformat_base": "reformat_base_methods",
    "reformat_clipped": "reformat_clipped_methods",
    "reformat_literal": "reformat_literal_methods",
    "reformat": "reformat_window_methods",
    "redacted_literal": "redacted_literal_methods",
    "redacted": "redacted_window_methods",
}

DB_NAMES = {
    "java-top10": "java-large-top10",
    "java-top50": "java-small-top50"
}

PARTITIONS = {
    "train": "training",
    "val": "validation",
    "test": "test"
}

# Modify (If)
IMG_TYPE = "redacted"
DB_NAME = "java-top10"

IMG_MODE = "L"
IMG_EXT = ".png"
IMG_TRANS_SIZE = 1024  # 32

NUM_WORKER = 2
