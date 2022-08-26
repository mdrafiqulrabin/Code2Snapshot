#!/bin/bash

mC2VRoot="/scratch/deployment/code-path/code2vec/"
mDataDir="/scratch/data/java_method_raw/"
mType=("methods")
mDataset=("java-top10" "java-top50")

for type in "${mType[@]}"; do
  for dataset in "${mDataset[@]}"; do
    echo "${type}/${dataset}"
    cd ${mC2VRoot}
    source preprocess.sh ${mDataDir}/${type} ${dataset} &> "preprocess-${type}-${dataset}.txt"
  done
done
