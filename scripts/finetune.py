#!/usr/bin/env python3
import os
import argparse
import subprocess
import time
import json
from pathlib import Path

def run_data_preparation(input_file, output_dir, val_split=0.1):
    """Run the data preparation script to convert and split the dataset"""
    cmd = [
        "python", "scripts/prepare_finetune_data.py",
        "--input", input_file,
        "--output_dir", output_dir,
        "--val_split", str(val_split)
    ]
    
    print("Running data preparation command:")
    print(" ".join(cmd))
    
    process = subprocess.run(cmd, check=True)
    return process.returncode == 0

def run_fine_tuning(config_path):
    """Run the MLX-LM fine-tuning with the given config"""
    cmd = [
        "mlx_lm.lora",
        "-c", config_path
    ]
    
    print("\nStarting fine-tuning with command:")
    print(" ".join(cmd))
    print("\nFine-tuning in progress, this may take several hours...")
    
    # Start time for tracking duration
    start_time = time.time()
    
    process = subprocess.run(cmd, check=True)
    success = process.returncode == 0
    
    # Calculate and print duration
    duration = time.time() - start_time
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\nFine-tuning completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    return success

def test_model(model, adapter_path=None, prompts=None, trust_remote_code=True, eos_token="<|endoftext|>"):
    """Test the fine-tuned model with sample prompts"""
    if prompts is None:
        prompts = [
            "Question: What is the significance of the Bhagavad Gita in the Mahabharata?\nAnswer:",
            "Question: Who were the Pandavas and what challenges did they face?\nAnswer:",
            "Question: What moral lessons does the Mahabharata teach about dharma?\nAnswer:"
        ]
    
    print("\nTesting fine-tuned model with sample prompts:")
    
    for i, prompt in enumerate(prompts, 1):
        cmd = ["mlx_lm.generate", "--model", model]
        
        if adapter_path:
            cmd.extend(["--adapter-path", adapter_path])
        
        cmd.extend(["--prompt", prompt])
        
        if trust_remote_code:
            cmd.append("--trust-remote-code")
        
        if eos_token:
            cmd.extend(["--eos-token", eos_token])
            
        cmd.extend(["--max-tokens", "200"])
        
        print(f"\nTest {i}:")
        print(f"Prompt: {prompt}")
        print("Generating response...\n")
        
        process = subprocess.run(cmd, capture_output=True, text=True, check=True)
        response = process.stdout
        
        print(response)
        
def main():
    parser = argparse.ArgumentParser(description="Run the entire fine-tuning pipeline")
    parser.add_argument("--input", default="/Users/mihirdave/Documents/Mahābhārata/mahabharata-llm/MahabharataPipeline/data/final/mahabharata_qa.jsonl", 
                        help="Path to input JSONL file with question/answer pairs")
    parser.add_argument("--config", default="finetune/finetune_config.yaml", 
                        help="Path to fine-tuning config file")
    parser.add_argument("--skip_data_prep", action="store_true",
                        help="Skip data preparation step (if already done)")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip the training step (if already done)")
    parser.add_argument("--test", action="store_true",
                        help="Run model testing after training")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Fraction of data to use for validation")
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs("finetune/data", exist_ok=True)
    os.makedirs("finetune/adapters", exist_ok=True)
    
    # Make paths absolute if not already
    input_file = os.path.abspath(args.input)
    config_path = os.path.abspath(args.config)
    
    # Step 1: Data preparation
    if not args.skip_data_prep:
        print("\n=== Step 1: Preparing Fine-Tuning Dataset ===")
        success = run_data_preparation(input_file, "finetune/data", args.val_split)
        if not success:
            print("Data preparation failed. Exiting.")
            return
    else:
        print("\n=== Skipping data preparation as requested ===")
    
    # Step 2: Fine-tuning
    if not args.skip_training:
        print("\n=== Step 2: Fine-Tuning the Model ===")
        success = run_fine_tuning(config_path)
        if not success:
            print("Fine-tuning failed. Exiting.")
            return
    else:
        print("\n=== Skipping fine-tuning as requested ===")
    
    # Step 3: Testing (optional)
    if args.test:
        print("\n=== Step 3: Testing the Fine-Tuned Model ===")
        
        # Load config to get model and adapter path
        with open(config_path, 'r') as f:
            config = {}
            for line in f:
                if ':' in line and not line.strip().startswith('#'):
                    key, value = line.split(':', 1)
                    config[key.strip()] = value.strip()
        
        model = config.get('model', "mlx-community/Qwen3-30B-A3B-4bit")
        adapter_path = config.get('adapter_path', "finetune/adapters")
        
        # Find the latest adapter if it exists
        adapter_dir = Path(adapter_path)
        if adapter_dir.exists():
            adapters = list(adapter_dir.glob("adapter_*.safetensors"))
            if adapters:
                latest_adapter = max(adapters, key=lambda p: int(p.stem.split('_')[1]))
                adapter_path = str(latest_adapter.parent)
            else:
                adapter_path = str(adapter_dir)
        
        test_model(model, adapter_path)
    
    print("\n=== Fine-Tuning Pipeline Complete ===")

if __name__ == "__main__":
    main() 