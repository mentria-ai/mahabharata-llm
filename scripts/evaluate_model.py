#!/usr/bin/env python3
import os
import argparse
import subprocess
import json
import random
from pathlib import Path
from tqdm import tqdm
import time
import csv

def sample_test_questions(input_jsonl, output_file, num_samples=10, seed=42):
    """Sample test questions from the dataset for evaluation"""
    random.seed(seed)
    
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    data = []
    for line in lines:
        try:
            item = json.loads(line.strip())
            if 'question' in item and 'answer' in item:
                data.append(item)
        except json.JSONDecodeError:
            continue
    
    # Sample random questions
    if len(data) > num_samples:
        samples = random.sample(data, num_samples)
    else:
        samples = data
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save sampled test questions
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in samples:
            f.write(json.dumps(item) + '\n')
    
    print(f"Sampled {len(samples)} test questions and saved to {output_file}")
    return samples

def generate_response(model, adapter_path, question, max_tokens=200):
    """Generate a response from the model for a given question"""
    prompt = f"Question: {question}\nAnswer:"
    
    cmd = ["mlx_lm.generate", "--model", model, "--prompt", prompt,
           "--max-tokens", str(max_tokens), "--trust-remote-code",
           "--eos-token", "<|endoftext|>"]
    
    if adapter_path:
        cmd.extend(["--adapter-path", adapter_path])
    
    process = subprocess.run(cmd, capture_output=True, text=True)
    if process.returncode != 0:
        print(f"Error generating response: {process.stderr}")
        return None
    
    # The output includes the prompt, so we need to extract just the answer
    response = process.stdout.strip()
    if "Answer:" in response:
        answer = response.split("Answer:", 1)[1].strip()
    else:
        answer = response
    
    return answer

def run_evaluation(model, adapter_path, test_questions, output_csv):
    """Run evaluation comparing base model vs fine-tuned model"""
    results = []
    
    print(f"Evaluating model responses for {len(test_questions)} questions...")
    
    for i, item in enumerate(tqdm(test_questions)):
        question = item["question"]
        reference_answer = item["answer"]
        
        # Base model response (without adapter)
        print(f"\nQuestion {i+1}/{len(test_questions)}: {question}")
        print("Generating base model response...")
        base_start = time.time()
        base_response = generate_response(model, None, question)
        base_time = time.time() - base_start
        
        # Fine-tuned model response (with adapter)
        print("Generating fine-tuned model response...")
        ft_start = time.time()
        ft_response = generate_response(model, adapter_path, question)
        ft_time = time.time() - ft_start
        
        result = {
            "question": question,
            "reference_answer": reference_answer,
            "base_response": base_response,
            "ft_response": ft_response,
            "base_time": base_time,
            "ft_time": ft_time
        }
        
        results.append(result)
        
        # Display the responses
        print("\nReference answer:", reference_answer)
        print("\nBase model response:", base_response)
        print("\nFine-tuned model response:", ft_response)
        print("-" * 80)
    
    # Save results to CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["question", "reference_answer", "base_response", 
                                              "ft_response", "base_time", "ft_time"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Evaluation complete. Results saved to {output_csv}")
    
    # Print summary
    base_avg_time = sum(r["base_time"] for r in results) / len(results)
    ft_avg_time = sum(r["ft_time"] for r in results) / len(results)
    
    print("\nEvaluation Summary:")
    print(f"Base model average response time: {base_avg_time:.2f} seconds")
    print(f"Fine-tuned model average response time: {ft_avg_time:.2f} seconds")
    print(f"See {output_csv} for detailed results and manual quality assessment")

def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare base model vs fine-tuned model")
    parser.add_argument("--model", default="mlx-community/Qwen3-30B-A3B-4bit", 
                        help="Model name or path")
    parser.add_argument("--adapter_path", required=True, 
                        help="Path to the fine-tuned model adapter directory")
    parser.add_argument("--input_file", default="../data/final/mahabharata_qa.jsonl", 
                        help="Path to the input JSONL file with question/answer pairs")
    parser.add_argument("--test_file", default="../finetune/test_questions.jsonl", 
                        help="Path to save/load test questions")
    parser.add_argument("--output_csv", default="../finetune/evaluation_results.csv", 
                        help="Path to save evaluation results")
    parser.add_argument("--num_samples", type=int, default=5, 
                        help="Number of test questions to sample (default: 5)")
    parser.add_argument("--sample", action="store_true", 
                        help="Force re-sampling of test questions even if test file exists")
    
    args = parser.parse_args()
    
    # Sample test questions or load existing ones
    if not os.path.exists(args.test_file) or args.sample:
        test_questions = sample_test_questions(args.input_file, args.test_file, args.num_samples)
    else:
        # Load existing test questions
        test_questions = []
        with open(args.test_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    test_questions.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    continue
        print(f"Loaded {len(test_questions)} existing test questions from {args.test_file}")
    
    # Run the evaluation
    run_evaluation(args.model, args.adapter_path, test_questions, args.output_csv)

if __name__ == "__main__":
    main() 