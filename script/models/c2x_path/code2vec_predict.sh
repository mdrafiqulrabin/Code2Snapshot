#!/bin/bash

mC2VRoot="/scratch/deployment/code-path/code2vec/"
mType=("methods")
mDataset=("java-top10" "java-top50")

for type in "${mType[@]}"; do
  for dataset in "${mDataset[@]}"; do
    echo "${type}/${dataset}"
    test_p="${mC2VRoot}/data/${type}/${dataset}/${dataset}.test.c2v"
    model_p="${mC2VRoot}/models/${type}/${dataset}-model/saved_model_best"
    result_p="${mC2VRoot}/results/${type}/${dataset}-model/log_test_best.txt"
    mkdir -p "${mC2VRoot}/results/${type}/${dataset}-model/"
    cd $mC2VRoot
    python3 code2vec.py --load ${model_p} --test ${test_p}
    mv log.txt ${result_p}
  done
done
