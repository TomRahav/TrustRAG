import os
import re
import csv

# Define the log lines you want to extract
desired_patterns = {
    "total_topk_num": r"total_topk_num: (\d+)",
    "total_adv_num": r"total_adv_num: (\d+)",
    "success_injection_rate": r"Success injection rate in top 5 contents: (\d+\.\d+%)",
    "inference_time": r"Total run time for inference, removal and defense: (\d+\.\d+)",
    "correct_answer_pct": r"Correct Answer Percentage: (\d+\.\d+%)",
    "incorrect_answer_pct": r"Incorrect Answer Percentage: (\d+\.\d+%)",
    "false_removal_rate": r"False removal rate in true content contents: (\d+\.\d+%)",
    "success_removal_rate": r"Success removal rate in adv contents: (\d+\.\d+%)",
}

# Initialize a list to store the extracted data
extracted_data = []

# Traverse through the 'logs' directory and its subdirectories
for root, dirs, files in os.walk("/home/tom.rahav/TrustRAG/logs"):
    for file in files:
        if file.endswith(".log"):
            file_path = os.path.join(root, file)
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
                content = "".join(lines)
                data = {}

                # Extract experiment name from the line containing 'Experiment Name --'
                experiment_name = None
                for line in lines:
                    if "Experiment Name --" in line:
                        match = re.search(r"Experiment Name --\s*(.*)", line)
                        if match:
                            experiment_name = match.group(1).strip()
                        break

                # Extract components from the experiment name
                if experiment_name:
                    # Define regex patterns for each component
                    patterns = {
                        "dataset": r"dataset_([^-\s]+)",
                        "eval_model": r"retriver_([^-\s]+)",
                        "model": r"model_(?:true_|false_)?([^-\s]+)",
                        "attack": r"attack_([^-\s]+)",
                        "removal": r"removal_([^-\s]+)",
                        "defense": r"defend_([^-\s]+)",
                        "adv_a_position": r"adv_a_position_([^-\s]+)",
                        "score": r"-(dot|cos)",
                        "adv_per_query": r"adv_per_query(\d+)",
                        "M": r"-M(\d+)",
                        "repeat": r"Repeat(\d+)",
                        "no_questions": r"no_questions",
                    }
                    for key, pattern in patterns.items():
                        match = re.search(pattern, experiment_name)
                        if key == "no_questions":
                            data[key] = 1 if match else 0
                        else:
                            data[key] = match.group(1) if match else None
                else:
                    # If experiment name is not found, set components to None or 0
                    for key in [
                        "dataset",
                        "eval_model",
                        "model",
                        "score",
                        "attack",
                        "adv_a_position",
                        "removal",
                        "no_questions",
                        "defense",
                        "adv_per_query",
                        "M",
                        "repeat",
                    ]:
                        data[key] = None
                    data["no_questions"] = 0
                # Extract other desired patterns
                for key, pattern in desired_patterns.items():
                    match = re.search(pattern, content)
                    data[key] = match.group(1) if match else None

                extracted_data.append(data)

# Define the CSV file headers
headers = [
    "dataset",
    "eval_model",
    "model",
    "score",
    "attack",
    "adv_a_position",
    "removal",
    "no_questions",
    "defense",
    "adv_per_query",
    "M",
    "repeat",
] + list(desired_patterns.keys())

# Write the extracted data to a CSV file
with open(
    "/home/tom.rahav/TrustRAG/notebooks/extracted_log_data.csv",
    "w",
    newline="",
    encoding="utf-8",
) as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=headers)
    writer.writeheader()
    for row in extracted_data:
        writer.writerow(row)

print("Data extraction complete. Results saved to 'extracted_log_data.csv'.")
