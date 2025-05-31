#!/bin/bash

# Define arrays of parameters 
datasets=('nq' 'hotpotqa') # 'nq' 'hotpotqa' 'msmarco' 'mirage'
models_names=("meta-llama/Llama-3.1-8B-Instruct") # "mistralai/Mistral-Nemo-Instruct-2407" "meta-llama/Llama-3.1-8B-Instruct" "gpt-4o"
eval_models_names=("minilm" "roberta") # "contriever" "contriever-ms" "ance" "minilm" "mpnet" "roberta"
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
                for apq in "$advs_per_query[@]}"; do
                  echo $index
                  # Submit the job with the current set of parameters
                  sbatch --export=DATASET="$dataset",MODEL_NAME="$model_name",EVAL_MODEL_CODE="$eval_model_name",ATTACK="$attack",REMOVAL="$removal",DEFENSE="$defense",SCORE="$score",ADV_A_POSITION="$position",ADV_PER_QUERY="$apq" scripts/job_script.sh
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