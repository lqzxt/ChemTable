import json
import os
import argparse
from tqdm import tqdm
from LLM import call_LLM
from utils import create_prompt, extract_json, evaluate_answer
import threading
from concurrent.futures import ThreadPoolExecutor
from template import qa_prompt_base_image, qa_prompt_base_html, qa_prompt_base_hybrid
from dataset import ChemTableDataset

parser = argparse.ArgumentParser(description='Evaluate logical reasoning trend questions')
parser.add_argument('--qa_mode', type=str, choices=['image', 'html', 'hybrid'], default='hybrid',
                   help='QA mode: image(only image), html(only HTML), hybrid(image+HTML)')
parser.add_argument('--model', type=str, default=None,
                   help='Specify model to evaluate, if not specified evaluate all models')
parser.add_argument('--threads', type=int, default=20,
                   help='Number of processing threads')
args = parser.parse_args()

data_file = "data/qa_en/logical_reasoning_trend.jsonl"
image_dir = "data/img"
output_dir = f"res/logical_reasoning_trend/{args.qa_mode}"
qa_mode = args.qa_mode

os.makedirs(output_dir, exist_ok=True)

MODEL_LIST = [
    "gpt-4.1-2025-04-14",
]

if args.model:
    MODEL_LIST = [model for model in MODEL_LIST if model == args.model]

MAX_SAMPLES = 1000

html_data_dict = {}
if qa_mode == "html" or qa_mode == "hybrid":
    print(f"Loading HTML data...")
    chem_dataset = ChemTableDataset()
    data_list = chem_dataset.getDataList()
    for item in data_list:
        html_data_dict[f"{item['id']}.png"] = item["clear_table_html"]
    print(f"Successfully loaded {len(html_data_dict)} HTML data entries")

def process_questions(model_name, output_file, num_threads=10):
    results = []
    results_lock = threading.Lock()
    file_lock = threading.Lock()
    
    with open(data_file, 'r', encoding='utf-8') as f:
        qa_pairs = [json.loads(line) for line in f]
    
    qa_pairs = qa_pairs[:MAX_SAMPLES]
    
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    result = json.loads(line)
                    processed_ids.add(result["image_id"])
                    results.append(result)
                except json.JSONDecodeError:
                    continue
        print(f"Loaded {len(processed_ids)} processed results from file")
    
    qa_pairs_to_process = [qa for qa in qa_pairs if qa["id"] not in processed_ids]
    
    if not qa_pairs_to_process:
        print(f"All questions processed, total {len(results)} results")
        return results
    
    pbar = tqdm(total=len(qa_pairs_to_process), desc=f"Processing questions ({model_name})", ncols=100)
    
    def process_single_question(qa_pair):
        try:
            question = qa_pair["question"]
            ground_truth = qa_pair["answer"]
            image_id = qa_pair["id"]
            unable_to_answer = qa_pair["unable_to_answer"]
            
            image_path = os.path.join(image_dir, image_id)
            
            if not os.path.exists(image_path):
                print(f"Image not found: {image_path}")
                pbar.update(1)
                return None
            
            try:
                if qa_mode == "image":
                    prompt_template = qa_prompt_base_image
                    prompt_text = prompt_template.replace("{Question}", question)
                    prompt = create_prompt(prompt_text, image_path)
                elif qa_mode == "html":
                    prompt_template = qa_prompt_base_html
                    table_html = html_data_dict.get(image_id, "")
                    if not table_html:
                        print(f"HTML data not found: {image_id}")
                        pbar.update(1)
                        return None
                    prompt_text = prompt_template.replace("{Question}", question).replace("{Table_html}", table_html)
                    prompt = create_prompt(prompt_text)
                elif qa_mode == "hybrid":
                    prompt_template = qa_prompt_base_hybrid
                    table_html = html_data_dict.get(image_id, "")
                    if not table_html:
                        print(f"HTML data not found: {image_id}")
                        pbar.update(1)
                        return None
                    prompt_text = prompt_template.replace("{Question}", question).replace("{Table_html}", table_html)
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
                "unable_to_answer": unable_to_answer
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
        executor.map(process_single_question, qa_pairs_to_process)
    
    pbar.close()
    
    return results

def run_evaluation_for_model(model_name, num_threads=10):
    model_file_name = model_name.replace('-', '_').replace('.', '_')
    output_file = os.path.join(output_dir, f"res_{model_file_name}_{qa_mode}.jsonl")
    
    if not os.path.exists(output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            pass
    
    print(f"Starting evaluation for model: {model_name} (QA mode: {qa_mode})")
    results = process_questions(model_name=model_name, output_file=output_file, num_threads=num_threads)
    
    total = len(results)
    correct = sum(1 for r in results if r["correctness"] == "correct")
    accuracy = correct / total if total > 0 else 0
    print(f"Model {model_name} accuracy: {accuracy:.2%} ({correct}/{total})")
    
    answerable_results = [r for r in results if not r["unable_to_answer"]]
    unanswerable_results = [r for r in results if r["unable_to_answer"]]
    
    answerable_total = len(answerable_results)
    answerable_correct = sum(1 for r in answerable_results if r["correctness"] == "correct")
    answerable_accuracy = answerable_correct / answerable_total if answerable_total > 0 else 0
    
    unanswerable_total = len(unanswerable_results)
    unanswerable_correct = sum(1 for r in unanswerable_results if r["correctness"] == "correct")
    unanswerable_accuracy = unanswerable_correct / unanswerable_total if unanswerable_total > 0 else 0
    
    print(f"Answerable questions accuracy: {answerable_accuracy:.2%} ({answerable_correct}/{answerable_total})")
    print(f"Unanswerable questions accuracy: {unanswerable_accuracy:.2%} ({unanswerable_correct}/{unanswerable_total})")
    
    summary_file = os.path.join(output_dir, f"summary_{qa_mode}.txt")
    with open(summary_file, 'a', encoding='utf-8') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Overall accuracy: {accuracy:.2%} ({correct}/{total})\n")
        f.write(f"Answerable questions accuracy: {answerable_accuracy:.2%} ({answerable_correct}/{answerable_total})\n")
        f.write(f"Unanswerable questions accuracy: {unanswerable_accuracy:.2%} ({unanswerable_correct}/{unanswerable_total})\n")
        f.write("-" * 50 + "\n")
    
    return results

def analyze_question_types():
    with open(data_file, 'r', encoding='utf-8') as f:
        qa_pairs = [json.loads(line) for line in f]
    
    qa_pairs = qa_pairs[:MAX_SAMPLES]
    
    categories = {
        "yield": [],
        "trend": [],
        "column": [],
        "increase": [],
        "decrease": []
    }
    
    for qa in qa_pairs:
        question = qa["question"].lower()
        for key in categories:
            if key in question:
                categories[key].append(qa)
    
    print("\nQuestion Type Analysis:")
    for category, questions in categories.items():
        print(f"{category}: {len(questions)} questions ({len(questions)/len(qa_pairs):.2%})")
    
    for model_name in MODEL_LIST:
        model_file_name = model_name.replace('-', '_').replace('.', '_')
        output_file = os.path.join(output_dir, f"res_{model_file_name}_{qa_mode}.jsonl")
        
        if not os.path.exists(output_file):
            continue
        
        with open(output_file, 'r', encoding='utf-8') as f:
            results = [json.loads(line) for line in f]
        
        results_dict = {r["image_id"]: r for r in results}
        
        print(f"\nModel {model_name} performance by question type (QA mode: {qa_mode}):")
        for category, questions in categories.items():
            category_results = [results_dict.get(q["id"]) for q in questions if results_dict.get(q["id"]) is not None]
            correct = sum(1 for r in category_results if r["correctness"] == "correct")
            accuracy = correct / len(category_results) if category_results else 0
            print(f"{category}: {accuracy:.2%} ({correct}/{len(category_results)})")
            
        summary_file = os.path.join(output_dir, f"summary_{qa_mode}.txt")
        with open(summary_file, 'a', encoding='utf-8') as f:
            f.write(f"\nModel {model_name} performance by question type:\n")
            for category, questions in categories.items():
                category_results = [results_dict.get(q["id"]) for q in questions if results_dict.get(q["id"]) is not None]
                correct = sum(1 for r in category_results if r["correctness"] == "correct")
                accuracy = correct / len(category_results) if category_results else 0
                f.write(f"{category}: {accuracy:.2%} ({correct}/{len(category_results)})\n")

if __name__ == "__main__":
    summary_file = os.path.join(output_dir, f"summary_{qa_mode}.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Logical Reasoning Trend Evaluation Summary (QA mode: {qa_mode})\n")
        f.write("=" * 50 + "\n\n")
    
    print(f"Starting evaluation with {qa_mode} mode...")
    
    for model_name in MODEL_LIST:
        try:
            run_evaluation_for_model(model_name, num_threads=args.threads)
        except Exception as e:
            print(f"Error processing model {model_name}: {str(e)}")
            continue
    
    analyze_question_types()
    
    print(f"All model evaluations completed! (QA mode: {qa_mode})")