from tqdm import tqdm
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import os
import argparse

from LLM import call_LLM
from dataset import ChemTableDataset
from template import get_smiles
from utils import *

def process_smiles(item, llm_name, result_queue):
    for smiles in item["smiles"]:
        smiles_id = smiles["smiles_id"]
        smiles_image_path = smiles["smiles_image_path"]
        smiles_gt = smiles["smiles_gt"].replace("[#smiles#]", "")

        prompt = create_prompt(get_smiles, smiles_image_path)
        resp = call_LLM(prompt, model_name=llm_name)
        pre_smiles = extract_smiles_from_response(resp)
        score = calculate_tanimoto_similarity(smiles_gt, pre_smiles)
        res = {
            "index": item["id"],
            "smiles_id": smiles_id,
            "gt": smiles_gt,
            "pre": pre_smiles,
            "score": score,
            "llm_name": llm_name
        }
        result_queue.put(res)

def result_writer(result_queue):
    results_by_model = {}
    
    while True:
        res = result_queue.get()
        if res is None:
            break
        
        llm_name = res.pop("llm_name")
        result_file = f"res/smiles/res_{llm_name}.jsonl"
        
        with open(result_file, 'a', encoding="utf-8") as f:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')
            
        if llm_name not in results_by_model:
            results_by_model[llm_name] = []
        results_by_model[llm_name].append(res["score"])
        
        avg_score = sum(results_by_model[llm_name]) / len(results_by_model[llm_name])
        print(f"Model {llm_name} current average score: {avg_score:.4f}, processed: {len(results_by_model[llm_name])}")
        
        result_queue.task_done()

def get_processed_items(result_file):
    processed = set()
    if not os.path.exists(result_file):
        return processed
    
    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                processed.add((data['index'], data['smiles_id']))
            except json.JSONDecodeError:
                continue
    return processed

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate SMILES recognition capability of multiple LLM models')
    parser.add_argument('--models', nargs='+', default=[
        "intern_vl"
    ], help='List of LLM models to evaluate')
    parser.add_argument('--workers', type=int, default=10, help='Number of worker threads per model')
    parser.add_argument('--max_samples', type=int, default=1000, help='Maximum number of samples to evaluate')
    parser.add_argument('--resume', default=True, action='store_true', help='Resume from checkpoint')
    args = parser.parse_args()
    
    data_list = ChemTableDataset().getDataList()
    
    if args.max_samples is not None:
        data_list = data_list[:args.max_samples]
        print(f"Limiting evaluation to first {args.max_samples} samples")
    
    os.makedirs("res/smiles", exist_ok=True)
    
    if not args.resume:
        for model in args.models:
            result_file = f"res/smiles/res_{model}.jsonl"
            if os.path.exists(result_file):
                os.remove(result_file)
                print(f"Cleaned old result file: {result_file}")
    
    result_queue = Queue()
    
    writer_thread = threading.Thread(target=result_writer, args=(result_queue,))
    writer_thread.daemon = True
    writer_thread.start()
    
    for llm_name in args.models:
        print(f"Starting model: {llm_name}")
        
        processed_items = set()
        if args.resume:
            result_file = f"res/smiles/res_{llm_name}.jsonl"
            processed_items = get_processed_items(result_file)
            print(f"Model {llm_name} has processed {len(processed_items)} samples, resuming from checkpoint")
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            for item in data_list:
                if args.resume:
                    unprocessed_smiles = []
                    for smiles in item["smiles"]:
                        if (item["id"], smiles["smiles_id"]) not in processed_items:
                            unprocessed_smiles.append(smiles)
                    
                    if not unprocessed_smiles:
                        continue
                    
                    new_item = item.copy()
                    new_item["smiles"] = unprocessed_smiles
                    futures.append(executor.submit(process_smiles, new_item, llm_name, result_queue))
                else:
                    futures.append(executor.submit(process_smiles, item, llm_name, result_queue))
            
            for f in tqdm(futures, total=len(futures), desc=f"Processing {llm_name}"):
                f.result()
        
        print(f"Model {llm_name} processing completed")
    
    result_queue.put(None)
    writer_thread.join()
    
    print("All model evaluations completed, results saved to res/smiles/ directory")