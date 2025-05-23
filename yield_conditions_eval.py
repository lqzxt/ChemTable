import json
import os
from tqdm import tqdm
from LLM import call_LLM
from utils import create_prompt, extract_json, evaluate_answer
import threading
from concurrent.futures import ThreadPoolExecutor
from template import qa_prompt_base_image, qa_prompt_base_html, qa_prompt_base_hybrid
from dataset import ChemTableDataset

QA_MODE = "image"

data_file = "data/qa_en/yield_and_conditions.jsonl"
image_dir = "data/img"
output_dir = "res/yield_conditions" + "_" + QA_MODE

os.makedirs(output_dir, exist_ok=True)

MODEL_LIST = [
    "gpt-4.1-2025-04-14"
]

def process_questions(model_name, output_file, num_threads=10, resume=False):
    results = []
    results_lock = threading.Lock()
    file_lock = threading.Lock()
    
    dataset = ChemTableDataset()
    data_list = dataset.getDataList()
    html_dict = {item["id"]: item["clear_table_html"] for item in data_list}
    
    with open(data_file, 'r', encoding='utf-8') as f:
        qa_pairs = [json.loads(line) for line in f]
    
    processed_ids = set()
    processed_ids = set()
    if resume and os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    result = json.loads(line)
                    processed_ids.add(result['image_id'])
                    results.append(result)
                except:
                    pass
        print(f"Resuming from checkpoint, processed {len(processed_ids)} samples")
    
    if resume:
        qa_pairs = [qa for qa in qa_pairs if qa['id'] not in processed_ids]
    
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
                image_id_num = int(os.path.splitext(image_id)[0])
                
                if QA_MODE == "image":
                    prompt_template = qa_prompt_base_image
                    prompt_text = prompt_template.replace("{Question}", question)
                    prompt = create_prompt(prompt_text, image_path)
                elif QA_MODE == "html":
                    prompt_template = qa_prompt_base_html
                    table_html = html_dict.get(image_id_num, "")
                    prompt_text = prompt_template.replace("{Question}", question).replace("{Table_html}", table_html)
                    prompt = create_prompt(prompt_text)
                elif QA_MODE == "hybrid":
                    prompt_template = qa_prompt_base_hybrid
                    table_html = html_dict.get(image_id_num, "")
                    prompt_text = prompt_template.replace("{Question}", question).replace("{Table_html}", table_html)
                    prompt = create_prompt(prompt_text, image_path)
                else:
                    prompt_template = qa_prompt_base_image
                    prompt_text = prompt_template.replace("{Question}", question)
                    prompt = create_prompt(prompt_text, image_path)
                
                llm_response = call_LLM(prompt, model_name=model_name)
                
                try:
                    response_json = extract_json(llm_response)
                    model_answer = response_json.get("answer", "")
                    model_thought = response_json.get("chain_of_thought", "")
                except Exception as e:
                    print(f"JSON parsing error: {str(e)}")
                    model_answer = llm_response
                    model_thought = llm_response
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
                "aspect": aspect,
                "thought": model_thought,
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
    
    correct_count = sum(1 for r in results if r["correctness"] == "correct")
    incorrect_count = sum(1 for r in results if r["correctness"] == "incorrect")
    unknown_count = sum(1 for r in results if r["correctness"] == "unknown")
    total_count = len(results)
    
    yield2cond_results = [r for r in results if r["aspect"] == "yield2cond"]
    cond2yield_results = [r for r in results if r["aspect"] == "cond2yield"]
    
    yield2cond_correct = sum(1 for r in yield2cond_results if r["correctness"] == "correct")
    cond2yield_correct = sum(1 for r in cond2yield_results if r["correctness"] == "correct")
    
    print(f"\nModel {model_name} evaluation results:")
    print(f"Total questions: {total_count}")
    print(f"Correct: {correct_count} ({correct_count/total_count*100:.2f}%)")
    print(f"Incorrect: {incorrect_count} ({incorrect_count/total_count*100:.2f}%)")
    print(f"Unknown: {unknown_count} ({unknown_count/total_count*100:.2f}%)")
    
    if yield2cond_results:
        print(f"\nyield2cond accuracy: {yield2cond_correct}/{len(yield2cond_results)} ({yield2cond_correct/len(yield2cond_results)*100:.2f}%)")
    
    if cond2yield_results:
        print(f"cond2yield accuracy: {cond2yield_correct}/{len(cond2yield_results)} ({cond2yield_correct/len(cond2yield_results)*100:.2f}%)")
    
    stats_file = os.path.join(output_dir, f"stats_{model_name.replace('-', '_').replace('.', '_')}.json")
    stats = {
        "model_name": model_name,
        "qa_mode": QA_MODE,
        "total_count": total_count,
        "correct_count": correct_count,
        "incorrect_count": incorrect_count,
        "unknown_count": unknown_count,
        "accuracy": correct_count/total_count if total_count > 0 else 0,
        "yield2cond_count": len(yield2cond_results),
        "yield2cond_correct": yield2cond_correct,
        "yield2cond_accuracy": yield2cond_correct/len(yield2cond_results) if yield2cond_results else 0,
        "cond2yield_count": len(cond2yield_results),
        "cond2yield_correct": cond2yield_correct,
        "cond2yield_accuracy": cond2yield_correct/len(cond2yield_results) if cond2yield_results else 0
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    return results

def run_evaluation_for_model(model_name, num_threads=10, resume=False):
    model_file_name = model_name.replace('-', '_').replace('.', '_')
    output_file = os.path.join(output_dir, f"res_{model_file_name}_{QA_MODE}.jsonl")
    
    if not resume:
        with open(output_file, 'w', encoding='utf-8') as f:
            pass
    
    print(f"Starting evaluation for model: {model_name}, QA mode: {QA_MODE}")
    results = process_questions(model_name=model_name, output_file=output_file, 
                               num_threads=num_threads, resume=resume)

    return results

def analyze_results(model_name=None):
    if model_name:
        model_file_name = model_name.replace('-', '_').replace('.', '_')
        stats_file = os.path.join(output_dir, f"stats_{model_file_name}.json")
        if os.path.exists(stats_file):
            with open(stats_file, 'r', encoding='utf-8') as f:
                stats = json.load(f)
                print(f"\nModel {stats['model_name']} evaluation results (QA mode: {stats.get('qa_mode', 'unknown')}):")
                print(f"Total accuracy: {stats['accuracy']*100:.2f}%")
                print(f"yield2cond accuracy: {stats['yield2cond_accuracy']*100:.2f}%")
                print(f"cond2yield accuracy: {stats['cond2yield_accuracy']*100:.2f}%")
        else:
            print(f"Stats file not found for model {model_name}")
    else:
        all_stats = []
        for model_name in MODEL_LIST:
            model_file_name = model_name.replace('-', '_').replace('.', '_')
            stats_file = os.path.join(output_dir, f"stats_{model_file_name}.json")
            if os.path.exists(stats_file):
                with open(stats_file, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                    all_stats.append(stats)
        
        if all_stats:
            all_stats.sort(key=lambda x: x['accuracy'], reverse=True)
            
            print("\nAll model evaluation results summary (sorted by total accuracy):")
            print("=" * 80)
            print(f"{'Model Name':<20} {'QA Mode':<10} {'Total Acc':<10} {'yield2cond Acc':<15} {'cond2yield Acc':<15}")
            print("-" * 80)
            
            for stats in all_stats:
                qa_mode = stats.get('qa_mode', 'unknown')
                print(f"{stats['model_name']:<20} {qa_mode:<10} {stats['accuracy']*100:>8.2f}% {stats['yield2cond_accuracy']*100:>13.2f}% {stats['cond2yield_accuracy']*100:>13.2f}%")
        else:
            print("No stats files found for any models")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate model performance on yield_and_conditions dataset')
    parser.add_argument('--model', type=str, help='Specify model name to evaluate')
    parser.add_argument('--analyze', action='store_true', help='Analyze existing evaluation results')
    parser.add_argument('--threads', type=int, default=20, help='Number of threads')
    parser.add_argument('--resume', default=True, action='store_true', help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    if args.analyze:
        analyze_results(args.model)
    else:
        if args.model:
            if args.model in MODEL_LIST:
                run_evaluation_for_model(args.model, num_threads=args.threads, 
                                        resume=args.resume)
            else:
                print(f"Unknown model: {args.model}")
                print(f"Available models: {', '.join(MODEL_LIST)}")
        else:
            for model_name in MODEL_LIST:
                try:
                    run_evaluation_for_model(model_name, num_threads=args.threads, 
                                           resume=args.resume)
                except Exception as e:
                    print(f"Error processing model {model_name}: {str(e)}")
                    continue
            
            analyze_results()
            
            print("All model evaluations completed!")