import json
import os
from tqdm import tqdm
from LLM import call_LLM
from utils import create_prompt, extract_json
import threading
from concurrent.futures import ThreadPoolExecutor
from utils import evaluate_answer
from template import qa_prompt_base_image

data_file = "data/qa_en/visual_reasoning.jsonl"
image_dir = "data/img"
output_dir = "res/visual_reasoning"

os.makedirs(output_dir, exist_ok=True)

MODEL_LIST = [
    "intern_vl",
]

def process_questions(model_name, output_file, num_threads=10):
    results = []
    results_lock = threading.Lock()
    file_lock = threading.Lock()
    
    with open(data_file, 'r', encoding='utf-8') as f:
        qa_pairs = [json.loads(line) for line in f]
    
    
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    result = json.loads(line)
                    results.append(result)
                    processed_ids.add(result['image_id'])
                except json.JSONDecodeError:
                    continue
        print(f"Loaded {len(results)} processed results")
    
    qa_pairs = [qa for qa in qa_pairs if qa['id'] not in processed_ids]
    
    if not qa_pairs:
        print("All questions processed")
        return results
    
    pbar = tqdm(total=len(qa_pairs), desc=f"Processing questions ({model_name})", ncols=100)
    
    def process_single_question(qa_pair):
        try:
            question = qa_pair["question"]
            ground_truth = qa_pair["answer"]
            image_id = qa_pair["id"]
            unable_to_answer = qa_pair["unable_to_answer"]
            aspect = qa_pair["aspect"]
            
            image_path = os.path.join(image_dir, image_id)
            
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                pbar.update(1)
                return None
            
            try:
                prompt_template = qa_prompt_base_image
                prompt_text = prompt_template.replace("{Question}", question)
                prompt = create_prompt(prompt_text, image_path)
                
                llm_response = call_LLM(prompt, model_name=model_name)
                
                try:
                    response_json = extract_json(llm_response)
                    model_answer = response_json.get("answer", "")
                except Exception as e:
                    print(f"JSON parsing error: {str(e)}")
                    model_answer = llm_response
            except Exception as e:
                print(f"Error processing question: {str(e)}")
                pbar.update(1)
                return None
            
            try:
                correctness = evaluate_answer(question, ground_truth, model_answer)
            except Exception as e:
                print(f"Error evaluating answer: {str(e)}")
                correctness = "unknown"
            
            result = {
                "question": question,
                "ground_truth": ground_truth,
                "model_answer": model_answer,
                "correctness": correctness,
                "image_id": image_id,
                "unable_to_answer": unable_to_answer,
                "aspect": aspect
            }
            
            with results_lock:
                results.append(result)
            
            with file_lock:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
        except Exception as e:
            print(f"Error processing question: {str(e)}")
        finally:
            pbar.update(1)
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(process_single_question, qa_pairs)
    
    pbar.close()
    
    return results

def run_evaluation_for_model(model_name, num_threads=10):
    model_file_name = model_name.replace('-', '_').replace('.', '_')
    output_file = os.path.join(output_dir, f"res_{model_file_name}.jsonl")
    
    if not os.path.exists(output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            pass
    
    print(f"Starting evaluation for model: {model_name}")
    results = process_questions(model_name=model_name, output_file=output_file, num_threads=num_threads)

    total = len(results)
    if total > 0:
        correct_count = sum(1 for r in results if r['correctness'] == 'correct')
        incorrect_count = sum(1 for r in results if r['correctness'] == 'incorrect')
        unknown_count = sum(1 for r in results if r['correctness'] == 'unknown')
        
        aspect_stats = {}
        for aspect in set(r['aspect'] for r in results):
            aspect_results = [r for r in results if r['aspect'] == aspect]
            aspect_total = len(aspect_results)
            aspect_correct = sum(1 for r in aspect_results if r['correctness'] == 'correct')
            aspect_stats[aspect] = {
                'total': aspect_total,
                'correct': aspect_correct,
                'accuracy': aspect_correct / aspect_total if aspect_total > 0 else 0
            }
        
        print(f"Model {model_name} evaluation results:")
        print(f"Total questions: {total}")
        print(f"Correct: {correct_count} ({correct_count/total*100:.2f}%)")
        print(f"Incorrect: {incorrect_count} ({incorrect_count/total*100:.2f}%)")
        print(f"Unknown: {unknown_count} ({unknown_count/total*100:.2f}%)")
        
        print("\nAccuracy by aspect:")
        for aspect, stats in aspect_stats.items():
            print(f"{aspect}: {stats['correct']}/{stats['total']} ({stats['accuracy']*100:.2f}%)")
        
        stats_file = os.path.join(output_dir, f"stats_{model_file_name}.json")
        stats = {
            'model': model_name,
            'total': total,
            'correct': correct_count,
            'incorrect': incorrect_count,
            'unknown': unknown_count,
            'accuracy': correct_count / total if total > 0 else 0,
            'aspect_stats': aspect_stats
        }
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

    return results

def analyze_results():
    models_stats = []
    
    for model_name in MODEL_LIST:
        model_file_name = model_name.replace('-', '_').replace('.', '_')
        stats_file = os.path.join(output_dir, f"stats_{model_file_name}.json")
        
        if os.path.exists(stats_file):
            with open(stats_file, 'r', encoding='utf-8') as f:
                stats = json.load(f)
                models_stats.append(stats)
    
    if not models_stats:
        print("No model statistics found")
        return
    
    report_file = os.path.join(output_dir, "comparison_report.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Visual Reasoning Evaluation Comparison Report\n\n")
        
        f.write("## Overall Accuracy\n\n")
        f.write("| Model | Total | Correct | Accuracy | Incorrect | Error Rate | Unknown | Unknown Rate |\n")
        f.write("|-------|-------|---------|----------|-----------|------------|---------|--------------|\n")
        
        for stats in sorted(models_stats, key=lambda x: x['accuracy'], reverse=True):
            model = stats['model']
            total = stats['total']
            correct = stats['correct']
            incorrect = stats['incorrect']
            unknown = stats['unknown']
            
            f.write(f"| {model} | {total} | {correct} | {correct/total*100:.2f}% | {incorrect} | {incorrect/total*100:.2f}% | {unknown} | {unknown/total*100:.2f}% |\n")
        
        f.write("\n## Accuracy by Aspect\n\n")
        
        all_aspects = set()
        for stats in models_stats:
            all_aspects.update(stats['aspect_stats'].keys())
        
        f.write("| Model | " + " | ".join(sorted(all_aspects)) + " |\n")
        f.write("|-------|" + "|".join(["------|" for _ in all_aspects]) + "\n")
        
        for stats in sorted(models_stats, key=lambda x: x['accuracy'], reverse=True):
            model = stats['model']
            aspect_stats = stats['aspect_stats']
            
            row = f"| {model} |"
            for aspect in sorted(all_aspects):
                if aspect in aspect_stats:
                    accuracy = aspect_stats[aspect]['accuracy'] * 100
                    row += f" {accuracy:.2f}% |"
                else:
                    row += " N/A |"
            
            f.write(row + "\n")
    
    print(f"Comparison report generated: {report_file}")


if __name__ == "__main__":
    for model_name in MODEL_LIST:
        try:
            run_evaluation_for_model(model_name, num_threads=20)
        except Exception as e:
            print(f"Error processing model {model_name}: {str(e)}")
            continue
    
    analyze_results()
    
    print("All model evaluations completed!")