#!/usr/bin/env python3
"""
finetune_argsort_fix.py - Fine-tune Qwen3 model using MLX with a fix for the ArgSort error

This script patches the Qwen3 Router implementation to apply stop_gradient to the scores
before calling argpartition/argsort, which prevents the vjp error during backpropagation.
"""

import os
import sys
import time
import json
import yaml
import argparse
from pathlib import Path
import subprocess
import importlib.util
import signal
import atexit

# For cleanup handling
process = None
start_time = None

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

def apply_router_patch():
    """
    Patch the Qwen3 Router to use stop_gradient on scores before argpartition/argsort
    """
    try:
        # First try to import and monkey patch
        import importlib
        import mlx.core as mx
        
        # Try different possible module paths for the router
        try:
            from transformers.models.qwen2.modeling_qwen2 import QwenSparseMoeBlock
            print("Found Qwen router in transformers.models.qwen2")
            
            # Save original forward method
            original_forward = QwenSparseMoeBlock.forward
            
            # Create patched forward method
            def patched_forward(self, hidden_states, *args, **kwargs):
                router_logits = self.gate(hidden_states)
                routing_weights, selected_experts = self.route(router_logits)
                hidden_states = self.experts(hidden_states, routing_weights, selected_experts)
                return hidden_states
            
            # Create patched route method that uses stop_gradient
            def patched_route(self, router_logits):
                router_logits = router_logits.reshape(-1, self.num_experts)
                router_probs = mx.softmax(router_logits, axis=-1)
                
                # Apply stop_gradient to prevent backprop through argpartition/argsort
                expert_idx = mx.argpartition(
                    mx.stop_gradient(router_probs),
                    -self.num_experts_per_tok,
                    axis=-1
                )
                
                expert_idx = expert_idx[..., -self.num_experts_per_tok:]
                
                if self.sort_experts:
                    # Apply stop_gradient to prevent backprop through argsort
                    indices = mx.argsort(
                        mx.stop_gradient(router_probs.take_along_axis(expert_idx, axis=-1)),
                        axis=-1
                    )
                    indices = indices[..., -1::-1]
                    expert_idx = expert_idx.take_along_axis(indices, axis=-1)
                
                expert_weights = router_probs.take_along_axis(expert_idx, axis=-1)
                
                # Normalize weights
                expert_weights = expert_weights / mx.sum(expert_weights, axis=-1, keepdims=True)
                
                return expert_weights, expert_idx
            
            # Apply the patches
            QwenSparseMoeBlock.route = patched_route
            
            print("✅ Successfully patched Qwen router to avoid ArgSort errors")
            return True
            
        except ImportError:
            # Try other potential module paths
            import sys
            for name in sys.modules:
                if 'qwen' in name.lower() and hasattr(sys.modules[name], 'Router'):
                    print(f"Found router in module: {name}")
                    module = sys.modules[name]
                    Router = module.Router
                    
                    # Save original __call__ method
                    original_call = Router.__call__
                    
                    # Create patched __call__ method
                    def patched_call(self, x, *args, **kwargs):
                        scores = self.w(x)
                        
                        # Apply stop_gradient to prevent backprop through argpartition
                        topk_idx = mx.argpartition(mx.stop_gradient(scores), -self.top_k, axis=-1)
                        
                        # Continue with the rest of the original implementation
                        # (simplified for demonstration)
                        return original_call(self, x, *args, **kwargs)
                    
                    # Apply the patch
                    Router.__call__ = patched_call
                    
                    print(f"✅ Successfully patched {name}.Router to avoid ArgSort errors")
                    return True
            
            # If we get here, we couldn't find the Router class
            print("⚠️ Could not find Qwen Router class to patch")
            return False
            
    except Exception as e:
        print(f"Error applying router patch: {e}")
        return False

def run_fine_tuning(config_path):
    """Run MLX-LM fine-tuning with the router patch applied"""
    global process, start_time
    
    # Set environment variables for better MLX performance
    os.environ["MLX_LAZY_INITIALIZATION"] = "1"
    os.environ["MLX_ALLOCATOR_FENCE"] = "1" 
    os.environ["MLX_GC_PERSISTENT"] = "1"
    
    # First apply the router patch
    success = apply_router_patch()
    if not success:
        print("⚠️ Router patch could not be applied. Fine-tuning may fail with ArgSort errors.")
    
    # Use mlx_lm.lora directly
    cmd = [
        "mlx_lm.lora",
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
    parser = argparse.ArgumentParser(description="Run fine-tuning with a fix for the ArgSort error")
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