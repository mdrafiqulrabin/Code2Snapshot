#!/bin/bash

mC2VRoot="/scratch/deployment/code-path/code2vec/"
mType=("methods")
mDataset=("java-top10" "java-top50")

for type in "${mType[@]}"; do
  for dataset in "${mDataset[@]}"; do
    echo "${type}/${dataset}"
    cd $mC2VRoot
    source train.sh "${type}" "${dataset}" &> "train-${type}-${dataset}.txt"
  done
done
