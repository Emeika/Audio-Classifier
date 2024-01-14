import os
import csv

def compare_csv_line_by_line(file1, file2):
    count_different_lines = 0
    total_lines = 0

    with open(file1, 'r') as f1, open(file2, 'r') as f2:
        reader1 = csv.reader(f1)
        reader2 = csv.reader(f2)

        header1 = next(reader1)
        header2 = next(reader2)

        if header1 != header2:
            print(f"Header mismatch between {file1} and {file2}")
            return 0, 0

        for row_num, (row1, row2) in enumerate(zip(reader1, reader2), start=1):
            total_lines += 1
            if row1 != row2:
                count_different_lines += 1

    return count_different_lines, total_lines

# List of CSV files to compare
csv_files = [
    # "results\submissions\Combination_1_Accuracy_predictions.csv",
    # "results\submissions\Combination_1_F1_Score_predictions.csv",
    # "results\submissions\Combination_2_Accuracy_predictions.csv",
    # "results\submissions\Combination_2_F1_Score_predictions.csv",
    # "results\submissions\Combination_3_Accuracy_predictions.csv",
    # "results\submissions\Combination_3_F1_Score_predictions.csv",
    # "results\submissions\Combination_4_Accuracy_predictions.csv",
    # "results\submissions\Combination_4_F1_Score_predictions.csv",
    "results\submissions\SVM, RandomForest, XGBoost_Combination_1_Accuracy_predictions.csv",
    "results\submissions\SVM, RandomForest, XGBoost_Combination_1_F1_Score_predictions.csv",
    "results\submissions\KNN, SVM, RandomForest_Combination_3_Accuracy_predictions.csv",
    "results\submissions\KNN, SVM, RandomForest_Combination_3_F1_Score_predictions.csv",
    "results\submissions\SVM, RandomForest, MLP_Combination_2_Accuracy_predictions.csv",
    "results\submissions\SVM, RandomForest, MLP_Combination_2_F1_Score_predictions.csv",
    "results\submissions\SVM, RandomForest, XGBoost, MLP_Combination_4_Accuracy_predictions.csv",
    "results\submissions\SVM, RandomForest, XGBoost, MLP_Combination_4_F1_Score_predictions.csv",
]

# Compare each CSV file with the first one line by line
first_file = csv_files[0]
file_name_without_path = os.path.basename(first_file)
count_different_lines = {file: compare_csv_line_by_line(first_file, file) for file in csv_files[1:]}

# Calculate percentage difference based on the count of different lines
total_lines = count_different_lines.get(file_name_without_path, (0, 0))[1]
print("\nPercentage differences from", first_file)
for file, (count_diff, total) in count_different_lines.items():
    percentage_difference = (count_diff / total) * 100 if total > 0 else 0
    print(f"{file}: {percentage_difference:.2f}% different")
