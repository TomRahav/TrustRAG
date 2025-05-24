#!/bin/bash

# Define arrays of parameters 
datasets=('nq' 'hotpotqa' 'msmarco') # 'nq' 'hotpotqa' 'msmarco'
models_names=("mistralai/Mistral-Nemo-Instruct-2407" "meta-llama/Llama-3.1-8B-Instruct") #  "mistralai/Mistral-Nemo-Instruct-2407" "meta-llama/Llama-3.1-8B-Instruct"
eval_models_names=("contriever") # "contriever" "contriever-ms" "ance"
attacks=("none") # "none" "LM_targeted" "hotflip" "pia"
removals=('drift' 'kmeans_ngram') # 'none' 'drift' 'kmeans' 'kmeans_ngram' 'all'
defenses=('none') # 'none' 'conflict' 'astute' 'instruct'
scores=('cos_sim') # 'dot' 'cos_sim'
positions=('end') # 'start' 'end'


index=1  # Initialize counter

# Iterate over each combination of parameters
for dataset in "${datasets[@]}"; do
  for model_name in "${models_names[@]}"; do
    for eval_model_name in "${eval_models_names[@]}"; do
      for attack in "${attacks[@]}"; do
        for removal in "${removals[@]}"; do
          for defense in "${defenses[@]}"; do
            for score in "${scores[@]}"; do
              for position in "${positions[@]}"; do
                echo $index
                # Submit the job with the current set of parameters
                sbatch --export=DATASET="$dataset",MODEL_NAME="$model_name",EVAL_MODEL_CODE="$eval_model_name",ATTACK="$attack",REMOVAL="$removal",DEFENSE="$defense",SCORE="$score",ADV_A_POSITION="$position" scripts/job_script.sh
                ((index++))  # Increment counter
                sleep 1  # Brief pause to be kind to the scheduler
              done
            done
          done
        done
      done
    done
  done
done