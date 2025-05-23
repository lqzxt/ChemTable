from tqdm import tqdm
from LLM import call_LLM
from dataset import ChemTableDataset
from template import *
from utils import *
from metric import TEDS
import concurrent.futures
import os
import json


def process_item(item, llm_name):
    try:
        gt_html = item["clear_table_html"]
        image_path = item["image_path"]
        prompt = create_prompt(tsr_html_prompt, image_path)
        resp = call_LLM(prompt, model_name=llm_name)
        try:
            pre_html = extract_HTML(resp)
        except Exception as e:
            print(e)
            print(resp)
            return None
        pre_html = format_td(pre_html)
        gt_html = format_td(gt_html)
        evaluator = TEDS(structure_only=False)
        evaluator_struct = TEDS(structure_only=True)
        TEDS_score = evaluator.evaluate(pre_html, gt_html)
        TEDS_Struct_score = evaluator_struct.evaluate(pre_html, gt_html)
        res = {
            "index": item["id"],
            "TEDS": TEDS_score,
            "TEDS_Struct": TEDS_Struct_score,
            "pre": pre_html,
            "gt": gt_html,
            "llm_name": llm_name
        }
        return res
    except Exception as e:
        print(f"Error processing item: {e}")
        return None


def save_result(result, llm_name):
    if result:
        with open(f"res/TR/res_{llm_name}.jsonl", 'a', encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()
            
            
def load_processed_items(llm_name):
    processed_ids = set()
    try:
        with open(f"res/TR/res_{llm_name}.jsonl", 'r', encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if "index" in item:
                        processed_ids.add(item["index"])
                except:
                    continue
    except FileNotFoundError:
        pass
    return processed_ids


if __name__ == '__main__':
    data_list = ChemTableDataset().getDataList()
    
    max_samples = 300
    if len(data_list) > max_samples:
        data_list = data_list[:max_samples]
        print(f"Limiting evaluation to first {max_samples} samples")
    
    llm_list = [
        "intern_vl",
    ]

    os.makedirs("res/TR", exist_ok=True)
    
    processed_items = {}
    for llm_name in llm_list:
        processed_items[llm_name] = load_processed_items(llm_name)
        print(f"{llm_name} has processed {len(processed_items[llm_name])} samples")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_info = {}
        for item in data_list:
            for llm_name in llm_list:
                if item["id"] in processed_items[llm_name]:
                    continue
                future = executor.submit(process_item, item, llm_name)
                future_to_info[future] = (item, llm_name)

        for future in tqdm(concurrent.futures.as_completed(future_to_info), total=len(future_to_info)):
            item, llm_name = future_to_info[future]
            result = future.result()
            save_result(result, llm_name)
            if result and "index" in result:
                processed_items[llm_name].add(result["index"])
