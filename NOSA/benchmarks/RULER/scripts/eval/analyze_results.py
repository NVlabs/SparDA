# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import sys

def analyze_results(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        return

    # The csv is in a transposed format where rows are [Tasks, Score, Nulls]
    # We want to extract scores and calculate averages
    
    # Extract task names and scores
    tasks = df.iloc[0, 1:].tolist()
    scores = pd.to_numeric(df.iloc[1, 1:], errors='coerce').tolist()
    
    # Create a more readable dataframe
    result_df = pd.DataFrame({'Task': tasks, 'Score': scores})
    
    # Calculate overall average
    overall_avg = result_df['Score'].mean()
    
    # Group by task categories (heuristic based on task names)
    categories = {
        'NIAH (Single)': [],
        'NIAH (Multikey)': [],
        'NIAH (Other)': [],
        'Variable Tracking': [],
        'Aggregation': [],
        'QA': []
    }
    
    for task, score in zip(tasks, scores):
        if 'niah_single' in task:
            categories['NIAH (Single)'].append(score)
        elif 'niah_multikey' in task:
            categories['NIAH (Multikey)'].append(score)
        elif 'niah' in task:
            categories['NIAH (Other)'].append(score)
        elif 'vt' in task:
            categories['Variable Tracking'].append(score)
        elif 'cwe' in task or 'fwe' in task:
            categories['Aggregation'].append(score)
        elif 'qa' in task:
            categories['QA'].append(score)
            
    print("\n" + "="*50)
    print("RULER Benchmark Results Analysis")
    print("="*50 + "\n")
    
    print(f"{'Task':<30} | {'Score':<10}")
    print("-" * 43)
    for index, row in result_df.iterrows():
        print(f"{row['Task']:<30} | {row['Score']:<10.2f}")
    print("-" * 43)
    print(f"{'OVERALL AVERAGE':<30} | {overall_avg:<10.2f}")
    print("\n" + "="*50 + "\n")

    print("Category Averages:")
    print("-" * 30)
    for cat, vals in categories.items():
        if vals:
            avg = sum(vals) / len(vals)
            print(f"{cat:<20}: {avg:.2f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_results(sys.argv[1])
    else:
        print("Usage: python analyze_results.py <path_to_summary.csv>")

