import os
import json
import argparse
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from template import qa_prompt_base_image
from utils import evaluate_answer, extract_json, encode_image
from LLM import call_LLM


def process_single_question(qa_item, images_dir, model_name, output_file, lock):
    id_value = qa_item.get('id')
    
    image_path = os.path.join(images_dir, id_value)
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None
    
    question = qa_item.get('question', '')
    ground_truth = qa_item.get('answer', '')
    
    unable_to_answer = qa_item.get('unable_to_answer', False)
    if unable_to_answer:
        print(f"Question {id_value} is marked as unable to answer, will verify if model correctly refuses to answer")
    
    try:
        prompt = qa_prompt_base_image.replace("{Question}", question)
        
        image_base64 = encode_image(image_path)
        
        print(f"Processing question {id_value}: {question}")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]
        
        response = call_LLM(messages, model_name=model_name)
        
        try:
            result = extract_json(response)
            model_answer = result.get('answer', '')
            correctness = evaluate_answer(question, ground_truth, model_answer)
            
            result_item = {
                'id': id_value,
                'question': question,
                'ground_truth': ground_truth,
                'model_answer': model_answer,
                'is_correct': correctness,
                'model': model_name,
                'unable_to_answer': unable_to_answer
            }
            
            with lock:
                with open(output_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(result_item, ensure_ascii=False) + '\n')
            
            is_correct = correctness.lower() == 'correct'
            return (id_value, is_correct, True)
            
        except Exception as e:
            print(f"Failed to parse response for question {id_value}: {e}")
            return None
            
    except Exception as e:
        print(f"Error processing question {id_value}: {e}")
        return None


def process_qa_file(file_path, images_dir, model_name, output_file, evaluated=None, id_range=None, num_threads=10, max_samples=None):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    if evaluated is None:
        evaluated = set()
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        evaluated.add(data.get('id'))
                    except json.JSONDecodeError:
                        continue
    
    total_questions = 0
    evaluated_questions = 0
    correct_answers = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        qa_data = [json.loads(line.strip()) for line in f if line.strip()]
    
    file_lock = threading.Lock()
    
    questions_to_process = []
    
    for qa_item in qa_data:
        id_value = qa_item.get('id')
        
        if id_range and id_value:
            try:
                record_id = int(id_value.split('.')[0])
                if record_id < id_range[0] or record_id > id_range[1]:
                    continue
            except (ValueError, IndexError):
                pass
        
        total_questions += 1
        
        if id_value in evaluated:
            evaluated_questions += 1
            continue
        
        questions_to_process.append(qa_item)
    
    if max_samples is not None:
        remaining_samples = max_samples - len(evaluated)
        if remaining_samples <= 0:
            print(f"Maximum sample limit reached ({max_samples}), no new samples will be processed")
            return {
                'model': model_name,
                'file': os.path.basename(file_path),
                'total_questions': total_questions,
                'evaluated_questions': len(evaluated),
                'correct_answers': 0,
                'accuracy': 0
            }
        
        if len(questions_to_process) > remaining_samples:
            print(f"Limiting sample processing to {remaining_samples} (total to process: {len(questions_to_process)})")
            questions_to_process = questions_to_process[:remaining_samples]
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for qa_item in questions_to_process:
            futures.append(
                executor.submit(
                    process_single_question, 
                    qa_item, 
                    images_dir, 
                    model_name, 
                    output_file, 
                    file_lock
                )
            )
        
        for future in futures:
            result = future.result()
            if result:
                id_value, is_correct, was_evaluated = result
                if was_evaluated:
                    evaluated_questions += 1
                    if is_correct:
                        correct_answers += 1
    
    if os.path.exists(output_file):
        correct_in_file = 0
        total_in_file = 0
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    total_in_file += 1
                    if data.get('is_correct', '').lower() == 'correct':
                        correct_in_file += 1
                except json.JSONDecodeError:
                    continue
        
        evaluated_questions = total_in_file
        correct_answers = correct_in_file
    
    accuracy = correct_answers / evaluated_questions if evaluated_questions > 0 else 0
    
    result_summary = {
        'model': model_name,
        'file': os.path.basename(file_path),
        'total_questions': total_questions,
        'evaluated_questions': evaluated_questions,
        'correct_answers': correct_answers,
        'accuracy': accuracy
    }
    
    print(f"\nEvaluation Results - {model_name} on {os.path.basename(file_path)}:")
    print(f"Total questions: {total_questions}")
    print(f"Evaluated questions: {evaluated_questions}")
    print(f"Correct answers: {correct_answers}")
    print(f"Accuracy: {accuracy:.4f}\n")
    
    return result_summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate large model table question answering capabilities')
    parser.add_argument('--models', nargs='+', 
                        default=['gemini-2.5-pro-preview-03-25'],
                        help='List of models to evaluate')
    parser.add_argument('--qa_dir', default='data/qa_en', help='QA dataset directory')
    parser.add_argument('--images_dir', default='data/img', help='Images directory')
    parser.add_argument('--output_dir', default='res/table_qa', help='Results output directory')
    parser.add_argument('--id_min', default=None, type=int, help='Minimum ID range value')
    parser.add_argument('--id_max', default=None, type=int, help='Maximum ID range value')
    parser.add_argument('--threads', default=30, type=int, help='Number of threads for parallel processing')
    parser.add_argument('--test_refusing', action='store_true', help='Test if model can correctly refuse to answer unanswerable questions')
    parser.add_argument('--max_samples', default=None, type=int, help='Maximum evaluation sample count')
    parser.add_argument('--resume', action='store_true', default=True, help='Continue from checkpoint')
    args = parser.parse_args()
    
    qa_files = [
        os.path.join(args.qa_dir, 'table_qa_position.jsonl')
    ]
    
    for file_path in qa_files:
        if not os.path.exists(file_path):
            print(f"QA data file does not exist: {file_path}")
            return
    
    id_range = None
    if args.id_min is not None and args.id_max is not None:
        id_range = (args.id_min, args.id_max)
        print(f"Will only evaluate records with IDs in range [{args.id_min}-{args.id_max}]")
    
    if args.max_samples is not None:
        print(f"Will limit each model to evaluate at most {args.max_samples} samples per dataset")
    
    print(f"Using {args.threads} threads for parallel evaluation")
    if args.test_refusing:
        print("Will test model's ability to refuse answering unanswerable questions")
    
    if args.resume:
        print("Will continue from checkpoint, reading existing evaluation results")
    
    all_results = defaultdict(list)
    
    for qa_file in qa_files:
        qa_type = os.path.basename(qa_file).replace('table_qa_', '').replace('.jsonl', '')
        
        dataset_dir = os.path.join(args.output_dir, qa_type)
        os.makedirs(dataset_dir, exist_ok=True)
        
        for model_name in args.models:
            output_file = os.path.join(dataset_dir, f"res_{model_name}.jsonl")
            
            evaluated = None
            if args.resume:
                evaluated = set()
                if os.path.exists(output_file):
                    with open(output_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                data = json.loads(line.strip())
                                evaluated.add(data.get('id'))
                            except json.JSONDecodeError:
                                continue
                    print(f"Read {len(evaluated)} previously evaluated question IDs from {output_file}")
            
            result = process_qa_file(
                qa_file, 
                args.images_dir, 
                model_name,
                output_file,
                evaluated=evaluated,
                id_range=id_range,
                num_threads=args.threads,
                max_samples=args.max_samples
            )
            
            all_results[model_name].append(result)
    
    print("\nOverall Model Performance:")
    for model_name, results in all_results.items():
        total_evaluated = sum(r['evaluated_questions'] for r in results)
        total_correct = sum(r['correct_answers'] for r in results)
        
        if total_evaluated > 0:
            overall_accuracy = total_correct / total_evaluated
            print(f"{model_name}: Accuracy = {overall_accuracy:.4f} ({total_correct}/{total_evaluated})")
            
            if args.test_refusing:
                unable_to_answer_count = 0
                correct_refusing_count = 0
                
                for qa_file in qa_files:
                    qa_type = os.path.basename(qa_file).replace('table_qa_', '').replace('.jsonl', '')
                    dataset_dir = os.path.join(args.output_dir, qa_type)
                    output_file = os.path.join(dataset_dir, f"res_{model_name}.jsonl")
                    
                    if os.path.exists(output_file):
                        with open(output_file, 'r', encoding='utf-8') as f:
                            for line in f:
                                try:
                                    data = json.loads(line.strip())
                                    if data.get('unable_to_answer', False):
                                        unable_to_answer_count += 1
                                        if data.get('is_correct', '').lower() == 'correct':
                                            correct_refusing_count += 1
                                except json.JSONDecodeError:
                                    continue
                
                if unable_to_answer_count > 0:
                    refusing_accuracy = correct_refusing_count / unable_to_answer_count
                    print(f"  Refusing accuracy for unanswerable questions = {refusing_accuracy:.4f} ({correct_refusing_count}/{unable_to_answer_count})")
        else:
            print(f"{model_name}: No questions evaluated")


if __name__ == "__main__":
    main() 