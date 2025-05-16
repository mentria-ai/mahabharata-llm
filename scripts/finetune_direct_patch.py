#!/usr/bin/env python3
"""
finetune_direct_patch.py - Fine-tune Qwen3 model with direct patching of the model code

This script directly patches the MLX router or model code to apply stop_gradient
to the scores before calling argpartition/argsort, without relying on imports.
"""

import os
import sys
import time
import json
import yaml
import argparse
from pathlib import Path
import subprocess
import signal
import atexit
import shutil
import tempfile
import importlib

# For cleanup handling
process = None
start_time = None
temp_dir = None

def cleanup_handler():
    """Handle cleanup when script is interrupted"""
    global process, temp_dir
    if process and process.poll() is None:
        print("\nCleaning up and terminating child process...")
        try:
            process.terminate()
            process.wait(timeout=5)
        except Exception as e:
            print(f"Failed to terminate process gracefully: {e}, trying to kill...")
            try:
                process.kill()
            except:
                pass
    
    # Remove temporary directory if it exists
    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Failed to clean up temporary directory: {e}")

def signal_handler(sig, frame):
    """Handle signals like Ctrl+C"""
    print(f"\nReceived signal {sig}, shutting down gracefully...")
    cleanup_handler()
    sys.exit(1)

def patch_model_loading():
    """
    Monkeypatch MLX-LM to apply stop_gradient before ArgSort/ArgPartition
    """
    try:
        # Create a temporary module with our patch
        patch_code = """
# Monkey patch mx.argpartition and mx.argsort within the scope of model loading
import mlx.core as mx
import types

# Save original functions
original_argpartition = mx.argpartition
original_argsort = mx.argsort

# Create patched versions
def patched_argpartition(a, k=None, axis=-1, stream=None, kth=None):
    # Handle either k or kth parameter (Qwen3 uses kth)
    if kth is not None:
        k = kth
    # Apply stop_gradient to prevent backprop through argpartition
    return original_argpartition(mx.stop_gradient(a), k, axis=axis, stream=stream)

def patched_argsort(a, axis=-1, stream=None):
    # Apply stop_gradient to prevent backprop through argsort
    return original_argsort(mx.stop_gradient(a), axis=axis, stream=stream)

# Apply patches
mx.argpartition = patched_argpartition
mx.argsort = patched_argsort

# This is just to confirm the patch was loaded
print("✅ Applied stop_gradient patch to mx.argpartition and mx.argsort")
"""
        
        # Write to a temporary file
        global temp_dir
        temp_dir = tempfile.mkdtemp()
        patch_file = os.path.join(temp_dir, "mlx_patch.py")
        
        with open(patch_file, "w") as f:
            f.write(patch_code)
        
        # Set environment variable to preload our patch
        old_pythonpath = os.environ.get('PYTHONPATH', '')
        if old_pythonpath:
            os.environ["PYTHONPATH"] = f"{temp_dir}:{old_pythonpath}"
        else:
            os.environ["PYTHONPATH"] = temp_dir
        
        # Create a preload environment variable to force Python to load our patch module
        os.environ["PYTHONSTARTUP"] = patch_file
        
        print("✅ Set up environment to patch MLX router operations")
        return True
    
    except Exception as e:
        print(f"Error setting up environment patches: {e}")
        return False

def run_fine_tuning(config_path):
    """Run MLX-LM fine-tuning with patched environment"""
    global process, start_time
    
    # Set environment variables for better MLX performance
    os.environ["MLX_LAZY_INITIALIZATION"] = "1"
    os.environ["MLX_ALLOCATOR_FENCE"] = "1" 
    os.environ["MLX_GC_PERSISTENT"] = "1"
    
    # Apply environment patches
    success = patch_model_loading()
    if not success:
        print("⚠️ Could not set up environment patches. Fine-tuning may fail.")
    
    # Create a custom wrapper script to ensure patches are loaded
    wrapper_script = os.path.join(temp_dir, "run_lora.py")
    with open(wrapper_script, "w") as f:
        f.write("""#!/usr/bin/env python3
import sys
import os
import mlx.core as mx

# Apply patches
original_argpartition = mx.argpartition
original_argsort = mx.argsort

def patched_argpartition(a, k=None, axis=-1, stream=None, kth=None):
    # Handle either k or kth parameter (Qwen3 uses kth)
    if kth is not None:
        k = kth
    # Apply stop_gradient to prevent backprop through argpartition
    return original_argpartition(mx.stop_gradient(a), k, axis=axis, stream=stream)

def patched_argsort(a, axis=-1, stream=None):
    # Apply stop_gradient to prevent backprop through argsort
    return original_argsort(mx.stop_gradient(a), axis=axis, stream=stream)

# Apply patches
mx.argpartition = patched_argpartition
mx.argsort = patched_argsort

print("✅ Applied stop_gradient patch to mx.argpartition and mx.argsort")

# Import and run mlx_lm.lora main function
from mlx_lm.lora import main
sys.exit(main())
""")
    
    # Make the wrapper script executable
    os.chmod(wrapper_script, 0o755)
    
    # Use our wrapper script
    cmd = [
        "python", wrapper_script,
        "-c", config_path,
    ]
    
    print("\nStarting fine-tuning with command:")
    print(" ".join(cmd))
    print("\nFine-tuning in progress, this may take several hours...")
    print("You can safely interrupt with Ctrl+C and resume later")
    
    # Start time for tracking duration
    start_time = time.time()
    
    # Use Popen to get more control over the process
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1  # Line buffered
        )
        
        # Real-time logging of output
        if process.stdout:
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

def main():
    parser = argparse.ArgumentParser(description="Run fine-tuning with direct patching of MLX functions")
    parser.add_argument("--config", default="finetune/finetune_config.yaml", 
                        help="Path to fine-tuning config file")
    
    args = parser.parse_args()
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup_handler)
    
    # Make path absolute if not already
    config_path = os.path.abspath(args.config)
    
    # Run fine-tuning with patch applied
    success = run_fine_tuning(config_path)
    
    if not success:
        print("Fine-tuning did not complete successfully.")
        
        # Check if there are any adapters saved
        adapter_dir = Path("finetune/adapters")
        adapters = list(adapter_dir.glob("adapter_*.safetensors"))
        if adapters:
            print(f"Found {len(adapters)} adapter checkpoint(s). Latest one:")
            latest_adapter = max(adapters, key=lambda p: int(p.stem.split('_')[1]))
            print(f"  - {latest_adapter}")
            print(f"You can test this checkpoint with: python scripts/chat.py --adapter_path finetune/adapters")

if __name__ == "__main__":
    main() 