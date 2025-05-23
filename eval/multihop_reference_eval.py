import json
import os
from tqdm import tqdm
from LLM import call_LLM
from utils import create_prompt, extract_json, evaluate_answer
import threading
from concurrent.futures import ThreadPoolExecutor
from template import qa_prompt_base_image
import argparse

data_file = "data/qa_en/multihop_reference.jsonl"
image_dir = "data/img"
output_dir = "res/multihop_reference"

os.makedirs(output_dir, exist_ok=True)

MODEL_LIST = [
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini-2025-04-14", 
    "claude-3-7-sonnet-20250219",
    "gemini-2.5-flash-preview-04-17",
    "qwen2.5-vl-72b-instruct"
]

def process_questions(model_name, output_file, num_threads=10, max_samples=None, resume=False):
    results = []
    results_lock = threading.Lock()
    file_lock = threading.Lock()
    
    with open(data_file, 'r', encoding='utf-8') as f:
        qa_pairs = [json.loads(line) for line in f]
    
    if max_samples and max_samples > 0:
        qa_pairs = qa_pairs[:max_samples]
    
    processed_ids = set()
    if resume and os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        result = json.loads(line)
                        processed_ids.add(result.get("image_id"))
                        results.append(result)
                    except:
                        pass
            print(f"Read {len(processed_ids)} processed results from file")
            
            qa_pairs = [qa for qa in qa_pairs if qa["id"] not in processed_ids]
            print(f"Remaining {len(qa_pairs)} questions to process")
        except Exception as e:
            print(f"Error reading existing results: {str(e)}")
    
    pbar = tqdm(total=len(qa_pairs), desc=f"Processing questions ({model_name})", ncols=100)
    
    total_questions = len(qa_pairs) + len(processed_ids)
    correct_answers = sum(1 for r in results if r.get("is_correct", False))
    correct_hopn = {2: 0, 3: 0, 4: 0}
    total_hopn = {2: 0, 3: 0, 4: 0}
    unable_to_answer_correct = 0
    total_unable_to_answer = 0
    
    for r in results:
        hop = r.get("hop", 2)
        if hop in total_hopn:
            total_hopn[hop] += 1
            if r.get("is_correct", False):
                correct_hopn[hop] += 1
        
        if r.get("unable_to_answer", False):
            total_unable_to_answer += 1
            if r.get("is_correct", False) and "unable to answer" in r.get("model_answer", "").lower():
                unable_to_answer_correct += 1
    
    def process_single_question(qa_pair):
        nonlocal correct_answers, total_questions
        nonlocal correct_hopn, total_hopn
        nonlocal unable_to_answer_correct, total_unable_to_answer
        
        try:
            question = qa_pair["question"]
            ground_truth = qa_pair["answer"]
            image_id = qa_pair["id"]
            unable_to_answer = qa_pair["unable_to_answer"]
            hop = qa_pair.get("hop", 2)
            
            if hop in total_hopn:
                total_hopn[hop] += 1
            
            if unable_to_answer:
                total_unable_to_answer += 1
            
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
                is_correct = correctness == "correct"
                
                if is_correct:
                    correct_answers += 1
                    if hop in correct_hopn:
                        correct_hopn[hop] += 1
                    
                    if unable_to_answer and "unable to answer" in model_answer.lower():
                        unable_to_answer_correct += 1
            except Exception as e:
                print(f"Error evaluating answer: {str(e)}")
                correctness = "unknown"
                is_correct = False
            
            result = {
                "question": question,
                "ground_truth": ground_truth,
                "model_answer": model_answer,
                "correctness": correctness,
                "is_correct": is_correct,
                "image_id": image_id,
                "unable_to_answer": unable_to_answer,
                "hop": hop
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
    
    accuracy = correct_answers / total_questions if total_questions > 0 else 0
    print(f"Model: {model_name}")
    print(f"Total accuracy: {accuracy:.4f} ({correct_answers}/{total_questions})")
    
    for hop in sorted(total_hopn.keys()):
        if total_hopn[hop] > 0:
            hop_accuracy = correct_hopn[hop] / total_hopn[hop]
            print(f"Hop {hop} accuracy: {hop_accuracy:.4f} ({correct_hopn[hop]}/{total_hopn[hop]})")
    
    unable_accuracy = unable_to_answer_correct / total_unable_to_answer if total_unable_to_answer > 0 else 0
    print(f"Unable to answer recognition accuracy: {unable_accuracy:.4f} ({unable_to_answer_correct}/{total_unable_to_answer})")
    
    stats = {
        "model_name": model_name,
        "total_accuracy": accuracy,
        "total_correct": correct_answers,
        "total_questions": total_questions,
        "hop_accuracy": {hop: correct_hopn[hop] / total_hopn[hop] if total_hopn[hop] > 0 else 0 for hop in total_hopn},
        "hop_correct": correct_hopn,
        "hop_total": total_hopn,
        "unable_to_answer_accuracy": unable_accuracy,
        "unable_to_answer_correct": unable_to_answer_correct,
        "total_unable_to_answer": total_unable_to_answer
    }
    
    stats_file = os.path.join(output_dir, f"stats_{model_name.replace('-', '_').replace('.', '_')}.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    return results

def run_evaluation_for_model(model_name, num_threads=10, max_samples=None, resume=False):
    model_file_name = model_name.replace('-', '_').replace('.', '_')
    output_file = os.path.join(output_dir, f"res_{model_file_name}.jsonl")
    
    if not resume and os.path.exists(output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            pass
    elif resume and not os.path.exists(output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            pass
    
    print(f"Starting evaluation for model: {model_name}")
    results = process_questions(model_name=model_name, output_file=output_file, 
                               num_threads=num_threads, max_samples=max_samples, resume=resume)

    return results

def analyze_results():
    all_stats = []
    
    for model_name in MODEL_LIST:
        model_file_name = model_name.replace('-', '_').replace('.', '_')
        stats_file = os.path.join(output_dir, f"stats_{model_file_name}.json")
        
        if os.path.exists(stats_file):
            with open(stats_file, 'r', encoding='utf-8') as f:
                stats = json.load(f)
                all_stats.append(stats)
    
    if not all_stats:
        print("No model statistics found.")
        return
    
    all_stats.sort(key=lambda x: x["total_accuracy"], reverse=True)
    
    print("\n===== Model Performance Comparison Report =====")
    print(f"{'Model Name':<30} {'Total Acc':<10} {'2-hop Acc':<10} {'3-hop Acc':<10} {'4-hop Acc':<10} {'Unable to Answer':<15}")
    
    for stats in all_stats:
        model_name = stats["model_name"]
        total_acc = f"{stats['total_accuracy']:.4f}"
        hop2_acc = f"{stats['hop_accuracy'].get('2', 0):.4f}"
        hop3_acc = f"{stats['hop_accuracy'].get('3', 0):.4f}"
        hop4_acc = f"{stats['hop_accuracy'].get('4', 0):.4f}"
        unable_acc = f"{stats['unable_to_answer_accuracy']:.4f}"
        
        print(f"{model_name:<30} {total_acc:<10} {hop2_acc:<10} {hop3_acc:<10} {hop4_acc:<10} {unable_acc:<15}")
    
    report_file = os.path.join(output_dir, "comparison_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    
    print(f"\nComparison report saved to: {report_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-hop Reference Evaluation Tool")
    parser.add_argument("--max_samples", type=int, default=None, 
                        help="Maximum number of samples to evaluate, default is all")
    parser.add_argument("--resume", default=True, action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--threads", type=int, default=10,
                        help="Number of threads, default is 10")
    parser.add_argument("--models", nargs="+", default=None,
                        help="List of models to evaluate, default is all models")
    args = parser.parse_args()
    
    models_to_evaluate = args.models if args.models else MODEL_LIST
    
    for model_name in models_to_evaluate:
        try:
            run_evaluation_for_model(model_name, num_threads=args.threads, 
                                    max_samples=args.max_samples, resume=args.resume)
        except Exception as e:
            print(f"Error processing model {model_name}: {str(e)}")
            continue
    
    analyze_results()
    
    print("All model evaluations completed!")