#!/bin/bash

model_list=(
  "all-mpnet-base-v2"
  "all-MiniLM-L12-v2"
  "all-MiniLM-L6-v2"
  "paraphrase-multilingual-MiniLM-L12-v2"
  "paraphrase-MiniLM-L3-v2"
)

k_values=(2 5 10)

for model in "${model_list[@]}"
do
  if [ -d "data/$model-index" ]; then
    echo "Directory data/$model-index already exists. Skipping index generation for model $model."
  else
    echo "Generating index for $model..."
    python scripts/generate-index.py -m "$model"
    echo "Index for model $model generated."
  fi

  echo "Running tests for model $model..."
  for k in "${k_values[@]}"
  do
    if [ -f "data/results/$model-$k.txt" ]; then
      echo "Result file data/results/$model-$k.txt already exists. Skipping testing with k=$k."
    else
      echo "Running test with k=$k..."
      python scripts/test-model.py -m "$model" -k "$k" >> "data/results/$model-$k.txt"
      echo "Test with k=$k completed."
    fi
  done
  echo "Tests completed for model $model."
done

