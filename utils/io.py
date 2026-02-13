import os
import csv


def save_results_csv(rows, output_csv, header):
    """
    Append result rows to a CSV file, writing the header only if the file is new.

    Args:
        rows: List of lists, each representing a CSV row.
        output_csv: Path to the output CSV file.
        header: List of column name strings.
    """
    file_exists = os.path.exists(output_csv)

    with open(output_csv, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        for row in rows:
            writer.writerow(row)
