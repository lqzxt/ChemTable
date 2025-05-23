#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import glob
from statistics import mean

def calculate_avg_scores(file_path):
    teds_scores = []
    teds_struct_scores = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                teds_scores.append(data.get('TEDS', 0))
                teds_struct_scores.append(data.get('TEDS_Struct', 0))
    teds_scores = teds_scores[:300]
    avg_teds = mean(teds_scores) if teds_scores else 0
    avg_teds_struct = mean(teds_struct_scores) if teds_struct_scores else 0
    count = len(teds_scores)
    
    return avg_teds, avg_teds_struct, count

def main():
    jsonl_files = glob.glob('res/TR/*.jsonl')
    
    print(f"{'Model Name':<40} {'Avg TEDS':<15} {'Avg TEDS-Struct':<15} {'Sample Count':<10}")
    print("-" * 80)
    
    for file_path in jsonl_files:
        model_name = os.path.basename(file_path).replace('res_', '').replace('.jsonl', '')
        avg_teds, avg_teds_struct, count = calculate_avg_scores(file_path)
        
        print(f"{model_name:<40} {avg_teds:.4f}       {avg_teds_struct:.4f}         {count}")

if __name__ == "__main__":
    main()
