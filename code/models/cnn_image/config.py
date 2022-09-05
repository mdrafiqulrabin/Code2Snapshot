ROOT_PATH = "/scratch/data/java_method_img/"

IMG_TYPES = {
    "original": "original_methods",
    "reformat_base": "reformat_base_methods",
    "reformat_clipped": "reformat_clipped_methods",
    "reformat_literal": "reformat_literal_methods",
    "reformat_window": "reformat_window_methods",
    "redacted_literal": "redacted_literal_methods",
    "redacted_window": "redacted_window_methods",
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

SIZE_PARAMS = {
    32: 576,
    64: 3136,
    128: 14400,
    256: 61504,
    512: 254016,
    1024: 1032256
}

# Modify (If)
IMG_TYPE = "redacted"
DB_NAME = "java-top10"

IMG_MODE = "L"
IMG_EXT = ".png"
IMG_TRANS_SIZE = 512

# Params
NUM_WORKER = 2
BATCH_SIZE = 64
MAX_EPOCH = 100
EARLY_LIMIT = 20
