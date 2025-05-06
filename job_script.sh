#!/bin/bash
#SBATCH --job-name=RAG_DEF
#SBATCH --output=logs/job_%j.txt    # Standard output
#SBATCH --error=logs/job_%j.txt      # Standard error
#SBATCH --ntasks=1                     # Number of tasks
#SBATCH -c 8                      # number of cores (treats)
#SBATCH --gres=gpu:A40:1
#SBATCH --mail-user=tom.rahav@campus.technion.ac.il
#SBATCH --mail-type=NONE                # Send email on all events

# Activate conda environment
conda init
conda activate trustrag

# Set constant parameters
gpu_id=0
seed=12
query_results_dir="main"
note=""

# Set parameters
DATASET="${DATASET:-hotpotqa}"           # ['nq','hotpotqa', 'msmarco']
MODEL_NAME="${MODEL_NAME:-mistralai/Mistral-Nemo-Instruct-2407}"
ATTACK="${ATTACK:-hotflip}"    # ['none', 'LM_targeted', 'hotflip', 'pia']
DEFENSE="${DEFENSE:-none}"       # ['none', 'conflict', 'astute', 'instruct']
REMOVAL="${REMOVAL:-none}"          # ['none', 'kmeans', 'kmeans_ngram']
SCORE="${SCORE:-dot}"

repeat_times=10
M=10                            # number of queries
eval_model_code="contriever"
split="test"
top_k=5
adv_per_query=3                # poison rate = adv_per_query / top_k
adv_a_position="end"
llm_flag=true
llm_arg=""
if $llm_flag; then
    llm_arg="--llm_flag"
fi

log_name="dataset_${DATASET}-retriver_${eval_model_code}-model_${llm_flag}_${MODEL_NAME}-M${M}xRepeat${repeat_times}-attack_${ATTACK}-removal_${REMOVAL}-defend_${DEFENSE}-${SCORE}-adv_per_query${adv_per_query}-Top_${top_k}-Seed_${seed}"
# Run the Python script with the parameters passed from the environment
python3 -u main_trustrag.py \
        --eval_dataset "$DATASET" \
        --model_name "$MODEL_NAME" \
        --attack_method "$ATTACK" \
        --defend_method "$DEFENSE" \
        --removal_method "$REMOVAL" \
        --score_function "$SCORE" \
        --eval_model_code "$eval_model_code" \
        --split "$split" \
        --query_results_dir "$query_results_dir" \
        --top_k "$top_k" \
        --gpu_id "$gpu_id" \
        --adv_per_query "$adv_per_query" \
        --repeat_times "$repeat_times" \
        --M "$M" \
        --seed "$seed" \
        --log_name "$log_name" \
        $llm_arg

