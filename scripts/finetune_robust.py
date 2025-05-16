#!/usr/bin/env python3
import os
import argparse
import subprocess
import time
import json
import sys
from pathlib import Path
import signal
import atexit

# Global variables for tracking
start_time = None
process = None

def cleanup_handler():
    """Handle cleanup when script is interrupted"""
    global process
    if process and process.poll() is None:
        print("\nCleaning up and terminating child process...")
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            print("Failed to terminate process gracefully, trying to kill...")
            try:
                process.kill()
            except:
                pass

def signal_handler(sig, frame):
    """Handle signals like Ctrl+C"""
    print(f"\nReceived signal {sig}, shutting down gracefully...")
    cleanup_handler()
    sys.exit(1)

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
    
    try:
        process = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Data preparation failed with error code {e.returncode}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error during data preparation: {str(e)}")
        return False

def run_fine_tuning(config_path):
    """Run the MLX-LM fine-tuning with the given config"""
    global process, start_time
    
    cmd = [
        "mlx_lm.lora",
        "-c", config_path,
        # Add any additional flags here if needed
    ]
    
    print("\nStarting fine-tuning with command:")
    print(" ".join(cmd))
    print("\nFine-tuning in progress, this may take several hours...")
    print("You can safely interrupt with Ctrl+C and resume later")
    
    # Start time for tracking duration
    start_time = time.time()
    
    # Use Popen instead of run to get more control
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1  # Line buffered
        )
        
        # Real-time logging of output
        if process.stdout:  # Check if stdout is not None
            for line in process.stdout:
                print(line, end='')
                sys.stdout.flush()
        else:
            print("Warning: Unable to capture process output")
        
        # Wait for process to complete
        process.wait()
        success = process.returncode == 0
        
        # Calculate and print duration
        duration = time.time() - start_time
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if success:
            print(f"\nFine-tuning completed successfully in {int(hours)}h {int(minutes)}m {int(seconds)}s")
        else:
            print(f"\nFine-tuning failed with exit code {process.returncode} after {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        return success
    
    except KeyboardInterrupt:
        print("\nFine-tuning interrupted by user")
        if process and process.poll() is None:
            print("Terminating fine-tuning process...")
            process.terminate()
            process.wait()
        
        duration = time.time() - start_time
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Process ran for {int(hours)}h {int(minutes)}m {int(seconds)}s before interruption")
        return False
    
    except Exception as e:
        print(f"Unexpected error during fine-tuning: {str(e)}")
        if process and process.poll() is None:
            process.terminate()
        return False

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
        
        try:
            process = subprocess.run(cmd, capture_output=True, text=True, check=True)
            response = process.stdout
            print(response)
        except subprocess.CalledProcessError as e:
            print(f"Error generating response: {e}")
            print(f"Error output: {e.stderr}")
            continue
        except Exception as e:
            print(f"Unexpected error during testing: {str(e)}")
            continue

def main():
    parser = argparse.ArgumentParser(description="Run the entire fine-tuning pipeline with robustness")
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
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup_handler)
    
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
            print("Fine-tuning did not complete successfully.")
            print("You can resume later with --skip_data_prep")
            
            # Check if there are any adapters saved
            adapter_dir = Path("finetune/adapters")
            adapters = list(adapter_dir.glob("adapter_*.safetensors"))
            if adapters:
                print(f"Found {len(adapters)} adapter checkpoint(s). Latest one:")
                latest_adapter = max(adapters, key=lambda p: int(p.stem.split('_')[1]))
                print(f"  - {latest_adapter}")
                print(f"You can test this checkpoint with: python scripts/chat.py --adapter_path finetune/adapters")
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
                print(f"Using latest adapter checkpoint from: {adapter_path}")
            else:
                adapter_path = str(adapter_dir)
                print(f"No adapter checkpoints found in {adapter_path}")
        
        test_model(model, adapter_path)
    
    print("\n=== Fine-Tuning Pipeline Complete ===")

if __name__ == "__main__":
    main() 