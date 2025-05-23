import warnings
import os
import json
from metric import *

def evaluate_jsonl_file(file_path, id_range=None):
    teds = TEDS()
    scores = []
    total_records = 0
    valid_scores = 0
    
    print(f"Processing file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                total_records += 1
                
                if id_range and 'id' in data:
                    try:
                        id_parts = data['id'].split('-')
                        if len(id_parts) > 0:
                            record_id = int(id_parts[0])
                            if record_id < id_range[0] or record_id > id_range[1]:
                                continue
                    except (ValueError, IndexError):
                        pass
                
                if data.get('score') is not None:
                    scores.append(data['score'])
                    valid_scores += 1
            except json.JSONDecodeError:
                print(f"Error: Unable to parse JSON line: {line}")
                continue
    
    id_range_info = f"ID range [{id_range[0]}-{id_range[1]}]" if id_range else ""
    print(f"  Processing complete: Total records {total_records}, {id_range_info} valid scores {valid_scores}")
    
    if valid_scores > 0:
        avg_score = sum(scores) / valid_scores
        return avg_score, valid_scores, total_records
    else:
        return 0, 0, total_records

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate average scores of SMILES records')
    parser.add_argument('--id-min', default=0, type=int, help='Minimum ID range')
    parser.add_argument('--id-max', default=50, type=int, help='Maximum ID range')
    args = parser.parse_args()
    
    id_range = None
    if args.id_min is not None and args.id_max is not None:
        id_range = (args.id_min, args.id_max)
        print(f"Will only calculate records with IDs in range [{args.id_min}-{args.id_max}]")
    
    smiles_dir = "res/smiles"
    all_files = [f for f in os.listdir(smiles_dir) if f.endswith('.jsonl')]
    
    if not all_files:
        print(f"No jsonl files found in {smiles_dir} directory")
        return
    
    print(f"Found {len(all_files)} jsonl files")
    for file_name in all_files:
        print(f"  - {file_name}")
    print("-" * 70)
    
    results = []
    
    for file_name in all_files:
        file_path = os.path.join(smiles_dir, file_name)
        avg_score, valid_scores, total_records = evaluate_jsonl_file(file_path, id_range)
        
        model_name = file_name.replace('res_', '').replace('.jsonl', '')
        
        results.append({
            'model': model_name,
            'file_name': file_name,
            'avg_score': avg_score,
            'valid_scores': valid_scores,
            'total_records': total_records,
            'coverage': valid_scores / total_records * 100 if total_records > 0 else 0
        })
    
    results.sort(key=lambda x: x['avg_score'], reverse=True)
    
    id_range_info = f"ID range [{id_range[0]}-{id_range[1]}]" if id_range else ""
    print(f"{'Model Name':<35} {id_range_info}{'Avg Score':<6} {'Coverage':<6} {'Valid/Total':<20}")
    print("-" * 70)
    
    total_avg_score = 0
    total_valid_scores = 0
    total_records = 0
    files_with_scores = 0
    
    for result in results:
        model_name = result['model']
        avg_score = result['avg_score']
        coverage = result['coverage']
        valid_scores = result['valid_scores']
        total = result['total_records']
        
        print(f"{model_name:<40} {avg_score:.4f}    {coverage:.2f}%     {valid_scores}/{total}")
        
        if avg_score > 0:
            total_avg_score += avg_score
            files_with_scores += 1
        
        total_valid_scores += valid_scores
        total_records += total
    
    print("-" * 70)
    if files_with_scores > 0:
        overall_avg = total_avg_score / files_with_scores
        overall_coverage = total_valid_scores / total_records * 100 if total_records > 0 else 0
        print(f"{'Overall Average':<40} {overall_avg:.4f}    {overall_coverage:.2f}%     {total_valid_scores}/{total_records}")
    else:
        print("No valid score records found")

if __name__ == "__main__":
    main()
