import json
import os
from tqdm import tqdm
from LLM import call_qwen_llm, call_LLM
from dataset import ChemTableDataset
from utils import create_prompt, extract_json, create_prompt_text
from template import *
from qa_answer_eval import evaluate_answer
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import time

qa_mode = "image"
llm_name = "llama-3.2-90b-vision-instruct"
data_file = "data/qa_en/statistic_qa_theEnd.jsonl"
output_file = f"res/statistic/res_{llm_name}_{qa_mode}.jsonl"
image_dir = "data/img"

os.makedirs(os.path.dirname(output_file), exist_ok=True)

def process_questions(limit=None, num_threads=50):
    results = []
    results_lock = threading.Lock()
    
    with open(data_file, 'r', encoding='utf-8') as f:
        qa_pairs = [json.loads(line) for line in f]
    
    if limit:
        qa_pairs = qa_pairs[:limit]
    
    pbar = tqdm(total=len(qa_pairs), desc="Processing questions", ncols=100)
    data_list = ChemTableDataset().getDataList()
    image_ids = {}
    for item in data_list:
        image_id = item["id"]
        image_ids[f"{image_id}.png"] = item['clear_table_html']
    for item in qa_pairs:
        key_ = f"{item['id']}"
        item["table_html"] = image_ids[key_]

    def process_single_question(qa_pair):
        try:
            question = qa_pair["question"]
            ground_truth = qa_pair["answer"]
            image_id = qa_pair["id"]
            category = qa_pair["category"]
            image_path = os.path.join(image_dir, image_id)
            
            if not os.path.exists(image_path):
                pbar.update(1)
                return None
            
            if qa_mode == "html":
                prompt_text = qa_prompt_base_html.replace("{Question}", question).replace("{Table_html}", qa_pair["table_html"])
                prompt = create_prompt_text(prompt_text)
            elif qa_mode == "hybrid":
                prompt_text = qa_prompt_base_hybrid.replace("{Question}", question).replace("{Table_html}", qa_pair["table_html"])
                prompt = create_prompt(prompt_text, image_path)
            else:
                prompt_text = qa_prompt_base_image.replace("{Question}", question)
                prompt = create_prompt(prompt_text, image_path)
            
            try:
                llm_response = call_LLM(prompt, model_name=llm_name)
                
                try:
                    response_json = extract_json(llm_response)
                    model_answer = response_json.get("answer", "")
                except Exception:
                    model_answer = llm_response
                
                correctness = evaluate_answer(question, ground_truth, model_answer)
                
                result = {
                    "question": question,
                    "ground_truth": ground_truth,
                    "model_answer": model_answer,
                    "correctness": correctness,
                    "image_id": image_id,
                    "category": category,
                    "qa_mode": qa_mode
                }
                
                with results_lock:
                    results.append(result)
                    if len(results) % 10 == 0:
                        save_results(results)
                
            except Exception as e:
                print(f"Error processing question: {str(e)}")
                
        except Exception as e:
            print(f"Error processing question: {str(e)}")
        finally:
            pbar.update(1)
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(process_single_question, qa_pairs)
    
    pbar.close()
    save_results(results)
    return results

def save_results(results):
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

def calculate_statistics(results):
    total = len(results)
    if total == 0:
        return
        
    correct = sum(1 for r in results if r["correctness"] == "correct")
    incorrect = sum(1 for r in results if r["correctness"] == "incorrect")
    unknown = sum(1 for r in results if r["correctness"] == "unknown")
    
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "correct": 0}
        categories[cat]["total"] += 1
        if r["correctness"] == "correct":
            categories[cat]["correct"] += 1

if __name__ == "__main__":
    results = process_questions(num_threads=30)
    calculate_statistics(results)