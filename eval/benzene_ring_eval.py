import json
import os
from tqdm import tqdm
from LLM import call_LLM
from utils import create_prompt, extract_json
import threading
from concurrent.futures import ThreadPoolExecutor
from utils import evaluate_answer
from template import qa_prompt_base_image
import argparse

data_file = "data/qa_en/benzene_ring_count.jsonl"
image_dir = "data/img"
output_dir = "res/benzene_ring"

os.makedirs(output_dir, exist_ok=True)

MODEL_LIST = [
    "intern_vl"
]

MAX_SAMPLES = 500

def process_questions(model_name, output_file, num_threads=10):
    results = []
    results_lock = threading.Lock()
    file_lock = threading.Lock()
    
    with open(data_file, 'r', encoding='utf-8') as f:
        qa_pairs = [json.loads(line) for line in f]
    
    qa_pairs = qa_pairs[:MAX_SAMPLES]
    
    evaluated_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    evaluated_ids.add(data.get('image_id'))
                except json.JSONDecodeError:
                    continue
        print(f"Found existing evaluation results, {len(evaluated_ids)} questions already evaluated")
    
    qa_pairs_to_process = [qa for qa in qa_pairs if qa["id"] not in evaluated_ids]
    
    pbar = tqdm(total=len(qa_pairs_to_process), desc=f"Processing questions ({model_name})", ncols=100)
    
    def process_single_question(qa_pair):
        try:
            question = qa_pair["question"]
            ground_truth = qa_pair["answer"]
            image_id = qa_pair["id"]
            unable_to_answer = qa_pair["unable_to_answer"]
            
            image_path = os.path.join(image_dir, image_id)
            
            if not os.path.exists(image_path):
                print(f"Image does not exist: {image_path}")
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
    output_file = os.path.join(output_dir, f"res_{model_file_name}.jsonl")
    
    if os.path.exists(output_file):
        print(f"Found existing result file: {output_file}")
        print(f"Will continue evaluation from checkpoint for model: {model_name}")
    else:
        with open(output_file, 'w', encoding='utf-8') as f:
            pass
        print(f"Starting evaluation for model: {model_name}")
    
    results = process_questions(model_name=model_name, output_file=output_file, num_threads=num_threads)

    if results:
        total = len(results)
        correct_count = sum(1 for r in results if r['correctness'] == 'correct')
        print(f"Processed {total} questions in this evaluation, accuracy: {correct_count/total*100:.2f}%")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model performance on benzene ring counting dataset')
    parser.add_argument('--model', type=str, help='Specify the model name to evaluate')
    parser.add_argument('--threads', type=int, default=10, help='Number of threads')
    
    args = parser.parse_args()
    
    if args.model:
        if args.model in MODEL_LIST:
            run_evaluation_for_model(args.model, num_threads=args.threads)
        else:
            print(f"Unknown model: {args.model}")
            print(f"Available models: {', '.join(MODEL_LIST)}")
    else:
        for model_name in MODEL_LIST:
            try:
                run_evaluation_for_model(model_name, num_threads=args.threads)
            except Exception as e:
                print(f"Error processing model {model_name}: {str(e)}")
                continue
    
    print("All model evaluations completed!")