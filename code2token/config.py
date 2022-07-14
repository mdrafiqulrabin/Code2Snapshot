ROOT_PATH = "/scratch/data/java_method_token/"

TOKEN_TYPES = {
    "value": "token_value",
    "kind": "token_kind",
    "xalnum": "token_xalnum",
    "literal": "method_literal",
    "xliteral": "method_xliteral"
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
TOKEN_TYPE = "value"
DB_NAME = "java-top10"

