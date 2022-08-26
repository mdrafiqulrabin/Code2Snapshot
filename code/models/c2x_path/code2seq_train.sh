#!/bin/bash

mC2SRoot="/scratch/deployment/code-path/code2seq/"
mType=("methods")
mDataset=("java-top10" "java-top50")

for type in "${mType[@]}"; do
  for dataset in "${mDataset[@]}"; do
    echo "${type}/${dataset}"
    cd $mC2SRoot
    source train.sh "${type}" "${dataset}" &> "train-${type}-${dataset}.txt"
  done
done
