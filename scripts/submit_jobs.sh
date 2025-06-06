#!/bin/bash

# Define arrays of parameters 
datasets=('msmarco') # 'nq' 'hotpotqa' 'msmarco' 'mirage'
models_names=("mistralai/Mistral-Nemo-Instruct-2407" "meta-llama/Llama-3.1-8B-Instruct" "gpt-4o") # "mistralai/Mistral-Nemo-Instruct-2407" "meta-llama/Llama-3.1-8B-Instruct" "gpt-4o"
eval_models_names=("contriever") # "contriever" "contriever-ms" "ance" "minilm" "mpnet" "roberta"
attacks=("hotflip") # "none" "LM_targeted" "hotflip" "pia"
removals=('drift' 'kmeans_ngram') # 'none' 'drift' 'kmeans' 'kmeans_ngram' 'all'
defenses=('none') # 'none' 'conflict' 'astute' 'instruct'
scores=('cos_sim') # 'dot' 'cos_sim'
positions=('start' 'end') # 'start' 'end'
advs_per_query=(1 3 5)


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
                for apq in "${advs_per_query[@]}"; do
                  echo $index
                  # Submit the job with the current set of parameters
                  sbatch --export=DATASET="$dataset",MODEL_NAME="$model_name",EVAL_MODEL_CODE="$eval_model_name",ATTACK="$attack",REMOVAL="$removal",DEFENSE="$defense",SCORE="$score",ADV_A_POSITION="$position",ADV_PER_QUERY="$apq" scripts/job_script.sh
                  # export DATASET="$dataset"
                  # export MODEL_NAME="$model_name"
                  # export EVAL_MODEL_CODE="$eval_model_name"
                  # export ATTACK="$attack"
                  # export REMOVAL="$removal"
                  # export DEFENSE="$defense"
                  # export SCORE="$score"
                  # export ADV_A_POSITION="$position"
                  # export ADV_PER_QUERY="$apq"
                  # bash scripts/job_script.sh  # Run the job script in the background
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
done