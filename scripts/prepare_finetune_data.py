#!/usr/bin/env python3
import json
import os
import random
import argparse
from tqdm import tqdm

def convert_qa_to_prompt_completion(input_file, output_train_file, output_valid_file, val_split=0.1, seed=42):
    """
    Convert the Q&A dataset to prompt/completion format and split into train/valid sets.
    
    Args:
        input_file: Path to the input JSONL file with question/answer pairs
        output_train_file: Path to save the training data
        output_valid_file: Path to save the validation data
        val_split: Fraction of data to use for validation (default: 0.1)
        seed: Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Loaded {len(lines)} question-answer pairs from {input_file}")
    
    # Parse and convert data
    converted_data = []
    for line in tqdm(lines, desc="Converting format"):
        try:
            data = json.loads(line.strip())
            # Format: convert question/answer to prompt/completion
            converted = {
                "prompt": f"Question: {data['question']}\nAnswer:",
                "completion": f" {data['answer']}"
            }
            converted_data.append(converted)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse line: {line[:50]}...")
            continue
        except KeyError:
            print(f"Warning: Missing question or answer key in: {line[:50]}...")
            continue
    
    # Shuffle the data
    random.shuffle(converted_data)
    
    # Split into train and validation sets
    split_idx = int(len(converted_data) * (1 - val_split))
    train_data = converted_data[:split_idx]
    valid_data = converted_data[split_idx:]
    
    print(f"Split data into {len(train_data)} training and {len(valid_data)} validation examples")
    
    # Create output directories if they don't exist
    os.makedirs(os.path.dirname(output_train_file), exist_ok=True)
    os.makedirs(os.path.dirname(output_valid_file), exist_ok=True)
    
    # Write train data
    with open(output_train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    # Write validation data
    with open(output_valid_file, 'w', encoding='utf-8') as f:
        for item in valid_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Saved training data to {output_train_file}")
    print(f"Saved validation data to {output_valid_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert Q&A dataset to MLX prompt/completion format")
    parser.add_argument("--input", default="../data/final/mahabharata_qa.jsonl", 
                        help="Path to input JSONL file with question/answer pairs")
    parser.add_argument("--output_dir", default="../finetune/data", 
                        help="Directory to save the training and validation data")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Fraction of data to use for validation (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Define output files
    output_train_file = os.path.join(args.output_dir, "train.jsonl")
    output_valid_file = os.path.join(args.output_dir, "valid.jsonl")
    
    convert_qa_to_prompt_completion(
        args.input, 
        output_train_file, 
        output_valid_file, 
        args.val_split, 
        args.seed
    )

if __name__ == "__main__":
    main() 