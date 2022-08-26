#!/bin/bash

mC2SRoot="/scratch/deployment/code-path/code2seq/"
mDataDir="/scratch/data/java_method_raw/"
mType=("methods")
mDataset=("java-top10" "java-top50")

for type in "${mType[@]}"; do
  for dataset in "${mDataset[@]}"; do
    echo "${type}/${dataset}"
    cd ${mC2SRoot}
    source preprocess.sh ${mDataDir}/${type} ${dataset} &> "preprocess-${type}-${dataset}.txt"
  done
done
