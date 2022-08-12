#!/usr/bin/env bash

db_names=("java-top10" "java-top50")
img_types=("original" "reformat" "redacted")

for db in "${db_names[@]}"; do
  for img in "${img_types[@]}"; do
    echo "${db} - ${img}"
    python3 sampleNet_train.py ${db} ${img} &> "/scratch/logs/sampleNet_${db}_${img}.log"
  done
done
