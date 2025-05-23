import json
import os
import glob
from statistics import mean

def calculate_avg_scores(file_path):
    title_scores = []
    anno_scores = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if data.get('title_score') is not None:
                    title_scores.append(data.get('title_score', 0))
                if data.get('anno_score') is not None:
                    anno_scores.append(data.get('anno_score', 0))
    
    avg_title = mean(title_scores) if title_scores else 0
    avg_anno = mean(anno_scores) if anno_scores else 0
    title_count = len(title_scores)
    anno_count = len(anno_scores)
    
    return avg_title, avg_anno, title_count, anno_count

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Calculate average scores for titles and annotations')
    parser.add_argument('--id-min', default=None, type=int, help='Minimum ID range')
    parser.add_argument('--id-max', default=None, type=int, help='Maximum ID range')
    args = parser.parse_args()
    
    if args.id_min is not None and args.id_max is not None:
        print(f"Only calculating records with IDs in range [{args.id_min}-{args.id_max}]")
    
    title_anno_dir = "res/title_anno"
    jsonl_files = glob.glob(f'{title_anno_dir}/*.jsonl')
    
    if not jsonl_files:
        print(f"No jsonl files found in {title_anno_dir} directory")
        return
    
    print(f"Found {len(jsonl_files)} jsonl files")
    for file_path in jsonl_files:
        print(f"  - {os.path.basename(file_path)}")
    print("-" * 80)
    
    results = []
    
    for file_path in jsonl_files:
        avg_title, avg_anno, title_count, anno_count = calculate_avg_scores(file_path)
        model_name = os.path.basename(file_path).replace('res_', '').replace('.jsonl', '')
        
        results.append({
            'model': model_name,
            'file_path': file_path,
            'avg_title': avg_title,
            'avg_anno': avg_anno,
            'title_count': title_count,
            'anno_count': anno_count
        })
    
    results.sort(key=lambda x: x['avg_title'], reverse=True)
    
    print(f"{'Model Name':<40} {'Title Score':<15} {'Annotation Score':<15} {'Title Count':<10} {'Annotation Count':<10}")
    print("-" * 100)
    
    total_title_score = 0
    total_anno_score = 0
    total_title_count = 0
    total_anno_count = 0
    files_with_scores = 0
    
    for result in results:
        model_name = result['model']
        avg_title = result['avg_title']
        avg_anno = result['avg_anno']
        title_count = result['title_count']
        anno_count = result['anno_count']
        
        print(f"{model_name:<40} {avg_title:.4f}       {avg_anno:.4f}         {title_count:<10} {anno_count:<10}")
        
        if avg_title > 0 or avg_anno > 0:
            if avg_title > 0:
                total_title_score += avg_title
            if avg_anno > 0:
                total_anno_score += avg_anno
            files_with_scores += 1
        
        total_title_count += title_count
        total_anno_count += anno_count
    
    print("-" * 100)
    if files_with_scores > 0:
        overall_title_avg = total_title_score / len([r for r in results if r['avg_title'] > 0]) if any(r['avg_title'] > 0 for r in results) else 0
        overall_anno_avg = total_anno_score / len([r for r in results if r['avg_anno'] > 0]) if any(r['avg_anno'] > 0 for r in results) else 0
        
        print(f"{'Overall Average':<40} {overall_title_avg:.4f}       {overall_anno_avg:.4f}         {total_title_count:<10} {total_anno_count:<10}")
    else:
        print("No valid score records found")

if __name__ == "__main__":
    main()