import os
import json
import pandas as pd
from collections import defaultdict

def analyze_results(base_dir='res/table_qa'):
    results = {}
    
    for task_name in os.listdir(base_dir):
        task_dir = os.path.join(base_dir, task_name)
        if not os.path.isdir(task_dir):
            continue
            
        task_results = {}
        
        for model_file in os.listdir(task_dir):
            if not model_file.endswith('.jsonl'):
                continue
                
            model_name = os.path.splitext(model_file)[0]
            model_path = os.path.join(task_dir, model_file)
            
            stats = defaultdict(int)
            total = 0
            total_unable = 0
            correct_unable = 0
            correct_not_unable = 0
            
            with open(model_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        
                        is_correct = data.get('is_correct', '')
                        unable_to_answer = data.get('unable_to_answer', False)
                        
                        if is_correct == 'unknown':
                            continue
                            
                        total += 1
                        
                        if unable_to_answer:
                            total_unable += 1
                            if is_correct == 'correct':
                                correct_unable += 1
                        else:
                            if is_correct == 'correct':
                                correct_not_unable += 1
                                
                        stats[is_correct] += 1
                        
                    except json.JSONDecodeError:
                        print(f"Error parsing line in {model_path}")
                        continue
            
            acc = stats['correct'] / total if total > 0 else 0
            acc_unable = correct_unable / total_unable if total_unable > 0 else 0
            total_not_unable = total - total_unable
            acc_unable_false = correct_not_unable / total_not_unable if total_not_unable > 0 else 0
            
            task_results[model_name] = {
                'ACC': acc,
                'ACC_unable': acc_unable,
                'ACC_unable_false': acc_unable_false,
                'total': total,
                'total_unable': total_unable,
                'correct_unable': correct_unable,
                'total_not_unable': total_not_unable,
                'correct_not_unable': correct_not_unable
            }
        
        results[task_name] = task_results
    
    return results

def format_results_as_tables(results):
    tables = {}
    
    for metric in ['ACC', 'ACC_unable', 'ACC_unable_false']:
        data = []
        
        for task_name, task_results in results.items():
            row = {'Task': task_name}
            
            for model_name, model_results in task_results.items():
                row[model_name] = model_results[metric]
            
            data.append(row)
        
        if data:
            df = pd.DataFrame(data)
            df = df.set_index('Task')
            tables[metric] = df
    
    return tables

def main():
    results = analyze_results()
    tables = format_results_as_tables(results)
    
    print("Experimental Results Statistics:")
    for metric, df in tables.items():
        print(f"\n{metric} Metrics:")
        print(df.to_string())
    
    print("\nDetailed Statistics:")
    for task_name, task_results in results.items():
        print(f"\nTask: {task_name}")
        for model_name, stats in task_results.items():
            print(f"  Model: {model_name}")
            for metric, value in stats.items():
                if isinstance(value, float):
                    print(f"    {metric}: {value:.4f}")
                else:
                    print(f"    {metric}: {value}")

if __name__ == "__main__":
    main()
