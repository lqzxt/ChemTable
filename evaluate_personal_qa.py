import json
import os
from tqdm import tqdm
from LLM import call_LLM
import threading
from concurrent.futures import ThreadPoolExecutor
from utils import extract_json, create_prompt
from template import *
from dataset import ChemTableDataset

QA_MODE = "hybrid"

MODELS_TO_EVALUATE = [
    "intern_vl",
]

input_file = "data/qa_en/personalization_questions_difficult_unique.jsonl"
output_dir = f"res/personal_{QA_MODE}"
model_verify = "gpt-4.1-nano-2025-04-14"
image_dir = "data/img"
html_dataset = None
num_threads = 20
limit_questions = None

os.makedirs(output_dir, exist_ok=True)

def get_qa_prompt():
    if QA_MODE == "html":
        return qa_prompt_base_html
    elif QA_MODE == "hybrid":
        return qa_prompt_base_hybrid
    else:
        return qa_prompt_base_image

answer_prompt = get_qa_prompt()
verify_prompt = qa_answer_eval

def load_html_dataset():
    global html_dataset
    if QA_MODE in ["html", "hybrid"] and html_dataset is None:
        print("Loading HTML dataset...")
        html_dataset = ChemTableDataset()
        print("HTML dataset loaded")

def process_questions(model_name, limit=None, num_threads=20):
    results = []
    stats = {"total": 0, "correct": 0, "incorrect": 0, "unknown": 0}
    results_lock = threading.Lock()
    
    if QA_MODE in ["html", "hybrid"]:
        load_html_dataset()
        html_map = {f"{item['id']}.png": item["clear_table_html"] for item in html_dataset.getDataList()}

    with open(input_file, 'r', encoding='utf-8') as f:
        qa_pairs = [json.loads(line) for line in f]
    
    if limit:
        qa_pairs = qa_pairs[:limit]
    
    pbar = tqdm(total=len(qa_pairs), desc=f"Evaluating model {model_name}", ncols=100)
    
    output_file = os.path.join(output_dir, f"res_{model_name.replace('-', '_').replace('.', '_')}.jsonl")
    
    def process_single_question(qa_pair):
        try:
            question = qa_pair["question"]
            ground_truth = qa_pair["answer"]
            image_id = qa_pair["id"]
            
            image_path = os.path.join(image_dir, image_id)
            
            if QA_MODE != "html" and not os.path.exists(image_path):
                pbar.update(1)
                return None
            
            if QA_MODE in ["html", "hybrid"] and image_id not in html_map:
                pbar.update(1)
                return None
            
            prompt_text = answer_prompt.replace("{Question}", question)
            
            if QA_MODE in ["html", "hybrid"]:
                prompt_text = prompt_text.replace("{Table_html}", html_map.get(image_id, ""))
            
            if QA_MODE == "html":
                prompt = [{"role": "user", "content": prompt_text}]
            else:
                prompt = create_prompt(prompt_text, image_path)
            
            model_answer_response = call_LLM(prompt, model_name=model_name)
            
            try:
                response_json = extract_json(model_answer_response)
                model_answer_text = response_json.get("answer", model_answer_response)
            except Exception:
                model_answer_text = model_answer_response
            
            verify_content = verify_prompt.replace("{Question}", question).replace("{Answer}", ground_truth).replace("{Model_Answer}", model_answer_text)
            verify_messages = [{"role": "user", "content": verify_content}]
            verify_response = call_LLM(verify_messages, model_name=model_verify)
            
            try:
                verification = extract_json(verify_response)
                if not verification:
                    verification = json.loads(verify_response)
                is_correct = verification.get("is_correct", "unknown")
                explanation = verification.get("chain_of_thought", "")
            except Exception as e:
                print(f"Error parsing verification response: {str(e)}")
                print(f"Original response: {verify_response}")
                is_correct = "unknown"
                explanation = "Parsing failed"
            
            result = {
                "id": qa_pair.get("id", ""),
                "question": question,
                "ground_truth": ground_truth,
                "model_answer": model_answer_text,
                "is_correct": is_correct,
                "verification_explanation": explanation
            }
            
            with results_lock:
                results.append(result)
                
                stats["total"] += 1
                if is_correct == "correct":
                    stats["correct"] += 1
                elif is_correct == "incorrect":
                    stats["incorrect"] += 1
                else:
                    stats["unknown"] += 1
                
                if len(results) % 10 == 0:
                    save_results(results, output_file)
                
        except Exception as e:
            print(f"Error processing question: {str(e)}")
        finally:
            pbar.update(1)
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(process_single_question, qa_pairs)
    
    pbar.close()
    
    save_results(results, output_file)
    return results, stats

def save_results(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

def analyze_results(results, stats, model_name):
    total = stats["total"]
    correct = stats["correct"]
    incorrect = stats["incorrect"]
    unknown = stats["unknown"]
    
    if total == 0:
        print("No questions processed")
        return
    
    accuracy = (correct / total) * 100
    
    print(f"\n{model_name} Evaluation Results:")
    print(f"QA Mode: {QA_MODE}")
    print(f"Total Questions: {total}")
    print(f"Correct Answers: {correct} ({accuracy:.2f}%)")
    print(f"Incorrect Answers: {incorrect} ({(incorrect / total) * 100:.2f}%)")
    print(f"Unknown Answers: {unknown} ({(unknown / total) * 100:.2f}%)")
    
    stats_file = os.path.join(output_dir, f"stats_{model_name.replace('-', '_').replace('.', '_')}.json")
    stats_data = {
        "qa_mode": QA_MODE,
        "model_name": model_name,
        "total_questions": total,
        "correct_answers": correct,
        "incorrect_answers": incorrect,
        "unknown_answers": unknown,
        "accuracy": correct / total if total > 0 else 0
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, ensure_ascii=False, indent=2)
    
    print(f"Statistics saved to {stats_file}")
    file_name = f"res_{model_name.replace('-', '_').replace('.', '_')}.jsonl"
    print(f"Detailed results saved to {os.path.join(output_dir, file_name)}")

def batch_evaluate_models(models, limit=None, num_threads=20):
    all_stats = {}
    
    print(f"\nBatch Evaluation Configuration:")
    print(f"QA Mode: {QA_MODE}")
    print(f"Threads: {num_threads}")
    print(f"Question Limit: {limit if limit is not None else 'No limit'}")
    
    for model_name in models:
        print(f"\nEvaluating model: {model_name}")
        results, stats = process_questions(model_name, limit=limit, num_threads=num_threads)
        analyze_results(results, stats, model_name)
        all_stats[model_name] = stats
    
    print("\nModel Comparison Results:")
    print("-" * 80)
    print(f"{'Model Name':<30} | {'Questions':<8} | {'Accuracy':<8} | {'Error Rate':<8} | {'Unknown Rate':<8}")
    print("-" * 80)
    
    for model_name, stats in all_stats.items():
        total = stats["total"]
        if total > 0:
            correct_rate = (stats["correct"] / total) * 100
            incorrect_rate = (stats["incorrect"] / total) * 100
            unknown_rate = (stats["unknown"] / total) * 100
            print(f"{model_name:<30} | {total:<8} | {correct_rate:<8.2f}% | {incorrect_rate:<8.2f}% | {unknown_rate:<8.2f}%")
    
    comparison_file = os.path.join(output_dir, "models_comparison.json")
    comparison_data = {
        "qa_mode": QA_MODE,
        "models": {
            model_name: {
                "total_questions": stats["total"],
                "correct_answers": stats["correct"],
                "incorrect_answers": stats["incorrect"],
                "unknown_answers": stats["unknown"],
                "accuracy": stats["correct"] / stats["total"] if stats["total"] > 0 else 0
            } for model_name, stats in all_stats.items()
        },
        "timestamp": __import__('datetime').datetime.now().isoformat()
    }
    
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison_data, f, ensure_ascii=False, indent=2)
    
    print(f"\nModel comparison report saved to {comparison_file}")

def run_default_evaluation():
    models_to_evaluate = MODELS_TO_EVALUATE
    limit = limit_questions
    threads = num_threads
    
    if not models_to_evaluate:
        print("No models found for evaluation. Please add models to MODELS_TO_EVALUATE at the top of the file")
        return
    
    print(f"Running evaluation with configuration from file header")
    print(f"QA Mode: {QA_MODE}")
    print(f"Models to evaluate: {', '.join(models_to_evaluate)}")
    print(f"Threads: {threads}")
    print(f"Question limit: {limit if limit is not None else 'No limit'}")
    print(f"Input file: {input_file}")
    
    if len(models_to_evaluate) == 1:
        print(f"Starting evaluation of model {models_to_evaluate[0]} on personalized questions...")
        results, stats = process_questions(models_to_evaluate[0], limit=limit, num_threads=threads)
        analyze_results(results, stats, models_to_evaluate[0])
    else:
        print(f"Starting batch evaluation of {len(models_to_evaluate)} models on personalized questions...")
        batch_evaluate_models(models_to_evaluate, limit=limit, num_threads=threads)

def main():
    import argparse
    
    models_to_evaluate = MODELS_TO_EVALUATE
    limit = limit_questions
    threads = num_threads
    global input_file
    global QA_MODE
    
    parser = argparse.ArgumentParser(description='Evaluate model accuracy on personalized questions')
    parser.add_argument('--model', type=str, help='Single model name to evaluate')
    parser.add_argument('--models', type=str, help='Multiple models to evaluate, comma-separated')
    parser.add_argument('--models-file', type=str, help='File containing list of models to evaluate, one per line')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of questions to process (for testing)')
    parser.add_argument('--threads', type=int, default=20, help='Number of threads to use')
    parser.add_argument('--input', type=str, default=input_file, help='Input data file path')
    parser.add_argument('--qa-mode', type=str, choices=['html', 'image', 'hybrid'], default=QA_MODE, 
                      help='QA mode: html, image, hybrid')
    parser.add_argument('--use-config', action='store_true', help='Ignore command line arguments, use file header config')
    
    args = parser.parse_args()
    
    if not args.use_config:
        if args.qa_mode != QA_MODE:
            QA_MODE = args.qa_mode
            global answer_prompt
            answer_prompt = get_qa_prompt()
            
        if args.input != input_file:
            input_file = args.input
            
        if args.threads != 20:
            threads = args.threads
            
        if args.limit is not None:
            limit = args.limit
            
        if args.model:
            models_to_evaluate = [args.model]
        elif args.models:
            models_to_evaluate = [model.strip() for model in args.models.split(',')]
        elif args.models_file:
            try:
                with open(args.models_file, 'r', encoding='utf-8') as f:
                    models_to_evaluate = [line.strip() for line in f if line.strip()]
            except Exception as e:
                print(f"Error reading model list file: {str(e)}")
                return
    
    if not models_to_evaluate:
        print("No models found for evaluation. Please add models to MODELS_TO_EVALUATE or specify via command line")
        return
    
    print(f"QA Mode: {QA_MODE}")
    print(f"Models to evaluate: {', '.join(models_to_evaluate)}")
    
    if len(models_to_evaluate) == 1:
        print(f"Starting evaluation of model {models_to_evaluate[0]} on personalized questions...")
        results, stats = process_questions(models_to_evaluate[0], limit=limit, num_threads=threads)
        analyze_results(results, stats, models_to_evaluate[0])
    else:
        print(f"Starting batch evaluation of {len(models_to_evaluate)} models on personalized questions...")
        batch_evaluate_models(models_to_evaluate, limit=limit, num_threads=threads)

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        run_default_evaluation()
    else:
        main()