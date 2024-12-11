
import csv

new_data = [
    ["Point", "Metric", "first_token", "last_token", "first_code_token", "last_code_token"],
    [32, "acc", 0.74, 0.66, 0.72, 0.75],
    [32, "f1-score", 0.74, 0.66, 0.72, 0.75],
    [28, "acc", 0.71, 0.67, 0.72, 0.74],
    [28, "f1-score", 0.70, 0.67, 0.72, 0.74],
    [24, "acc", 0.76, 0.66, 0.70, 0.74],
    [24, "f1-score", 0.76, 0.66, 0.70, 0.73],
    [20, "acc", 0.73, 0.68, 0.67, 0.71],
    [20, "f1-score", 0.73, 0.67, 0.67, 0.71],
    [16, "acc", 0.77, 0.68, 0.68, 0.78],
    [16, "f1-score", 0.77, 0.68, 0.68, 0.77],
    [12, "acc", 0.62, 0.62, 0.65, 0.70],
    [12, "f1-score", 0.62, 0.62, 0.65, 0.70],
    [8, "acc", 0.61, 0.60, 0.59, 0.61],
    [8, "f1-score", 0.60, 0.60, 0.58, 0.61],
    [4, "acc", 0.58, 0.62, 0.57, 0.60],
    [4, "f1-score", 0.58, 0.61, 0.57, 0.59],
    [1, "acc", 0.63, 0.58, 0.64, 0.60],
    [1, "f1-score", 0.62, 0.55, 0.63, 0.59],
]

# Prepare files for acc and f1-score based on the new data
new_metric_files = {}
for metric in ["acc", "f1-score"]:
    metric_data = [["Point", metric, "first_token", "last_token", "first_code_token", "last_code_token"]]
    metric_data += [
        [row[0], row[1], row[2], row[3], row[4], row[5]]
        for row in new_data[1:] if row[1] == metric
    ]
    file_path = f"/drive2/tuandung/WCODELLM/csv/{metric}_new_data.csv"
    with open(file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(metric_data)
    new_metric_files[metric] = file_path

new_metric_files