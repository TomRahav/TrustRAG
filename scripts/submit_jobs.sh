#!/bin/bash

# Define arrays of parameters 
datasets=('nq' 'hotpotqa' 'msmarco') # 'nq' 'hotpotqa' 'msmarco'
models_names=("meta-llama/Llama-3.1-8B-Instruct" "mistralai/Mistral-Nemo-Instruct-2407") #  "mistralai/Mistral-Nemo-Instruct-2407" "meta-llama/Llama-3.1-8B-Instruct"
attacks=("hotflip") # "none" "LM_targeted" "hotflip" "pia"
removals=('kmeans_ngram') # 'none' 'kmeans' 'kmeans_ngram'
defenses=('none') # 'none' 'conflict' 'astute' 'instruct'
scores=('dot') # 'dot' 'cos_sim'
positions=('start' 'end') # 'start' 'end'


index=1  # Initialize counter

# Iterate over each combination of parameters
for dataset in "${datasets[@]}"; do
  for model_name in "${models_names[@]}"; do
    for attack in "${attacks[@]}"; do
      for removal in "${removals[@]}"; do
        for defense in "${defenses[@]}"; do
          for score in "${scores[@]}"; do
            for position in "${positions[@]}"; do
              echo $index
              # Submit the job with the current set of parameters
              sbatch --export=DATASET="$dataset",MODEL_NAME="$model_name",ATTACK="$attack",REMOVAL="$removal",DEFENSE="$defense",SCORE="$score",ADV_A_POSITION="$position" scripts/job_script.sh
              ((index++))  # Increment counter
              sleep 1  # Brief pause to be kind to the scheduler
            done
          done
        done
      done
    done
  done
done