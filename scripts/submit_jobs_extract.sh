#!/bin/bash

# Define arrays of parameters 
datasets=('msmarco') # 'nq' 'hotpotqa' 'msmarco' 'mirage'
eval_models_names=("minilm" "mpnet" "roberta") # "contriever" "contriever-ms" "ance" "minilm" "mpnet" "roberta"
scores=('cos_sim') # 'dot' 'cos_sim'
topks=(20) # 5 10 20

index=1  # Initialize counter

# Iterate over each combination of parameters
for dataset in "${datasets[@]}"; do
  for eval_model_name in "${eval_models_names[@]}"; do
    for score in "${scores[@]}"; do
      for topk in "${topks[@]}"; do
        echo $index
        # Submit the job with the current set of parameters
        sbatch --export=DATASET="$dataset",EVAL_MODEL_CODE="$eval_model_name",SCORE="$score",TOPK="$topk" scripts/job_script_extract.sh
        ((index++))  # Increment counter
        sleep 1  # Brief pause to be kind to the scheduler
      done
    done
  done
done