#!/usr/bin/env bash

db_names=("java-top10" "java-top50")
token_types=("value" "kind" "xalnum" "literal" "xliteral")

for db in "${db_names[@]}"; do
  for type in "${token_types[@]}"; do
    echo "${db} - ${type}"
    python3 bilstm_train.py ${db} ${type} &> "/scratch/logs/bilstm_${db}_${type}.log"
  done
done
