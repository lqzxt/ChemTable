import json
import os
from collections import defaultdict

def calculate_category_accuracy(file_path):
    category_stats = defaultdict(lambda: {"correct": 0, "total": 0, "accuracy": 0.0})
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    result = json.loads(line)
                    category = result.get("category", "Uncategorized")
                    if category == "":
                        category = "Uncategorized"
                    
                    correctness = result.get("correctness", "")
                    
                    category_stats[category]["total"] += 1
                    if correctness == "correct":
                        category_stats[category]["correct"] += 1
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return {}
    
    total_correct = 0
    total_samples = 0
    
    for category, stats in category_stats.items():
        if stats["total"] > 0:
            stats["accuracy"] = (stats["correct"] / stats["total"]) * 100
        total_correct += stats["correct"]
        total_samples += stats["total"]
    
    overall_accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
    category_stats["Overall"] = {
        "correct": total_correct,
        "total": total_samples,
        "accuracy": overall_accuracy
    }
    
    return category_stats

def print_category_accuracy(category_stats):
    print("\nCategory Accuracy Statistics:")
    print("=" * 70)
    print(f"{'Category':<30} {'Correct':<10} {'Total':<10} {'Accuracy':<10}")
    print("-" * 70)
    
    overall_stats = category_stats.pop("Overall", None)
    if overall_stats:
        print(f"{'Overall':<30} {overall_stats['correct']:<10} {overall_stats['total']:<10} {overall_stats['accuracy']:.2f}%")
        print("-" * 70)
    
    sorted_categories = sorted(category_stats.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    
    for category, stats in sorted_categories:
        print(f"{category:<30} {stats['correct']:<10} {stats['total']:<10} {stats['accuracy']:.2f}%")
    
    print("=" * 70)
    
    if overall_stats:
        category_stats["Overall"] = overall_stats

def save_results_to_json(category_stats, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(category_stats, f, ensure_ascii=False, indent=2)
    print(f"Accuracy statistics saved to: {output_file}")

def main():
    result_file = "res/unable_to_answer/res_intern_vl.jsonl"
    output_dir = "res/unable_to_answer/accuracy_analysis"
    
    os.makedirs(output_dir, exist_ok=True)
    
    category_stats = calculate_category_accuracy(result_file)
    
    print_category_accuracy(category_stats)
    
    json_file = os.path.join(output_dir, "category_accuracy.json")
    save_results_to_json(category_stats, json_file)

if __name__ == "__main__":
    main()