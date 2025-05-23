import json
import os
import time
import base64
from tqdm import tqdm
from LLM import call_LLM
from template import qa_prompt_base_image, qa_answer_eval
from utils import extract_json

data_file = "data/qa_en/statistic_qa.jsonl"
output_file = "res/statistic_qa_results.jsonl"
image_dir = "data/img"

os.makedirs(os.path.dirname(output_file), exist_ok=True)

def get_image_type(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    if ext == '.jpg' or ext == '.jpeg':
        return 'jpeg'
    elif ext == '.png':
        return 'png'
    else:
        return 'jpeg'

def get_image_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
    except Exception as e:
        print(f"Failed to read image {image_path}: {e}")
        return None

def create_image_message(prompt, image_path):
    base64_image = get_image_base64(image_path)
    if not base64_image:
        return None
    
    return [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{get_image_type(image_path)};base64,{base64_image}"
                }
            }
        ]
    }]

def evaluate_answer(question, ground_truth, model_answer):
    prompt = qa_answer_eval.replace("{Question}", question).replace("{Answer}", ground_truth).replace("{Model_Answer}", model_answer)
    
    try:
        eval_response = call_LLM([{"role": "user", "content": prompt}], model_name="gpt-4")
        
        try:
            eval_result = extract_json(eval_response)
            return eval_result.get("is_correct", "unknown")
        except json.JSONDecodeError:
            print(f"Failed to parse evaluation result: {eval_response}")
            return "unknown"
    except Exception as e:
        print(f"Error evaluating answer: {e}")
        return "unknown"

def process_questions(limit=10):
    results = []
    
    with open(data_file, 'r', encoding='utf-8') as f:
        qa_pairs = [json.loads(line) for line in f]
    
    if limit > 0:
        qa_pairs = qa_pairs[:limit]
    
    print(f"Loaded {len(qa_pairs)} questions")
    
    for i, qa_pair in enumerate(tqdm(qa_pairs)):
        question = qa_pair["question"]
        ground_truth = qa_pair["answer"]
        image_id = qa_pair["id"]
        category = qa_pair["category"]
        
        image_path = os.path.join(image_dir, image_id)
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        
        prompt = qa_prompt_base_image.replace("{Question}", question)
        
        messages = create_image_message(prompt, image_path)
        if not messages:
            print(f"Failed to create image message for question {i+1}")
            continue
        
        try:
            print(f"Processing question {i+1}: {question}")
            
            llm_response = call_LLM(messages)
            
            try:
                response_json = json.loads(llm_response)
                model_answer = response_json.get("answer", "")
            except json.JSONDecodeError:
                model_answer = llm_response
                
            print(f"Model answer: {model_answer}")
            
            print(f"Evaluating answer...")
            correctness = evaluate_answer(question, ground_truth, model_answer)
            print(f"Correctness: {correctness}")
            
            result = {
                "question": question,
                "ground_truth": ground_truth,
                "model_answer": model_answer,
                "correctness": correctness,
                "image_id": image_id,
                "category": category
            }
            results.append(result)
            
            save_results(results)
                
            time.sleep(2)
            
        except Exception as e:
            print(f"Error processing question: {e}")
    
    save_results(results)
    return results

def save_results(results):
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    print(f"Saved {len(results)} results to {output_file}")

def calculate_statistics(results):
    total = len(results)
    if total == 0:
        print("No results to analyze")
        return
        
    correct = sum(1 for r in results if r["correctness"] == "correct")
    incorrect = sum(1 for r in results if r["correctness"] == "incorrect")
    unknown = sum(1 for r in results if r["correctness"] == "unknown")
    
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "correct": 0, "incorrect": 0, "unknown": 0}
        
        categories[cat]["total"] += 1
        if r["correctness"] == "correct":
            categories[cat]["correct"] += 1
        elif r["correctness"] == "incorrect":
            categories[cat]["incorrect"] += 1
        else:
            categories[cat]["unknown"] += 1
    
    print(f"Total questions: {total}")
    print(f"Correct: {correct} ({correct/total*100:.2f}%)")
    print(f"Incorrect: {incorrect} ({incorrect/total*100:.2f}%)")
    print(f"Unknown: {unknown} ({unknown/total*100:.2f}%)")
    print("\nCategory statistics:")
    
    for cat, stats in categories.items():
        correct_rate = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
        print(f"  {cat}: Total {stats['total']}, Accuracy {correct_rate:.2f}%")

if __name__ == "__main__":
    print("Starting question processing...")
    results = process_questions(limit=50)
    calculate_statistics(results)
    print("Processing completed!")
