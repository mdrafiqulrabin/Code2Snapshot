#!/bin/bash

mC2SRoot="/scratch/deployment/code-path/code2seq/"
mType=("methods")
mDataset=("java-top10" "java-top50")

for type in "${mType[@]}"; do
  for dataset in "${mDataset[@]}"; do
    echo "${type}/${dataset}"
    test_p="${mC2SRoot}/data/${type}/${dataset}/${dataset}.test.c2s"
    model_p="${mC2SRoot}/models/${type}/${dataset}-model/saved_model_best"
    result_p="${mC2SRoot}/results/${type}/${dataset}-model/log_test_best.txt"
    mkdir -p "${mC2SRoot}/results/${type}/${dataset}-model/"
    cd $mC2SRoot
    python3 code2seq.py --load ${model_p} --test ${test_p}
    mv log.txt ${result_p}
  done
done
