#!/usr/bin/env python3
import os
import argparse
import subprocess
import textwrap
import sys
import time

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def format_message(role, content):
    """Format a message based on its role"""
    if role == "assistant":
        return f"\033[92m{content}\033[0m"  # Green for assistant
    elif role == "user":
        return f"\033[96m{content}\033[0m"  # Cyan for user
    else:
        return f"\033[93m{content}\033[0m"  # Yellow for system

def print_header():
    """Print a header for the chat interface"""
    clear_screen()
    header = """
    🔮 Mahabharata Expert AI - Fine-tuned with MLX 🔮
    
    Ask me questions about the Mahabharata epic!
    Type 'exit' or 'quit' to end the conversation.
    --------------------------------------------------
    """
    print(header)

def wrap_text(text, width=80):
    """Wrap text to a specified width"""
    return '\n'.join(textwrap.wrap(text, width=width))

def generate_response(model, adapter_path, prompt, max_tokens=500, temperature=0.7):
    """Generate a response from the model"""
    # Start the command
    cmd = ["mlx_lm.generate", "--model", model]
    
    if adapter_path:
        cmd.extend(["--adapter-path", adapter_path])
    
    cmd.extend([
        "--prompt", prompt,
        "--max-tokens", str(max_tokens),
        "--temp", str(temperature),
        "--extra-eos-token", "<|endoftext|>"
    ])
    
    # Execute the command and capture output
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    if process.returncode != 0:
        print(f"Error generating response: {process.stderr}")
        return "I'm sorry, I encountered an error while processing your request."
    
    # Process the output to extract just the generated text
    response = process.stdout.strip()
    
    # Extract just the answer part if needed
    if "Answer:" in response:
        answer = response.split("Answer:", 1)[1].strip()
    else:
        answer = response.replace(prompt, "").strip()
    
    # Clean up the response
    answer = answer.replace("==========", "")
    answer = answer.replace("<think>", "")
    answer = answer.replace("</think>", "")
    
    # Remove MLX generation stats
    if "tokens-per-sec" in answer:
        answer = answer.split("Prompt:", 1)[0].strip()
    
    return answer

def interactive_chat(model, adapter_path, system_prompt=None, max_tokens=500, temperature=0.7):
    """Run an interactive chat session with the model"""
    print_header()
    
    # Display system prompt if provided
    if system_prompt:
        print(format_message("system", system_prompt))
        print()
    
    history = []
    
    while True:
        # Get user input
        user_input = input(format_message("user", "You: "))
        print()
        
        # Check for exit command
        if user_input.lower() in ["exit", "quit", "bye"]:
            print(format_message("system", "Thank you for chatting! Goodbye."))
            break
        
        # Format the prompt with the current question
        prompt = f"Question: {user_input}\nAnswer:"
        
        # Display thinking indicator
        print(format_message("system", "Thinking..."), end="\r")
        
        # Generate response
        start_time = time.time()
        response = generate_response(model, adapter_path, prompt, max_tokens, temperature)
        generation_time = time.time() - start_time
        
        # Clear the thinking indicator
        print(" " * 30, end="\r")
        
        # Add to history
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})
        
        # Print the response with word wrapping
        print(format_message("assistant", f"AI: {wrap_text(response)}"))
        print(format_message("system", f"[Response time: {generation_time:.2f}s]"))
        print()

def main():
    parser = argparse.ArgumentParser(description="Interactive chat with fine-tuned Mahabharata AI")
    parser.add_argument("--model", default="mlx-community/Qwen3-30B-A3B-4bit", 
                        help="Model name or path")
    parser.add_argument("--adapter_path", required=True, 
                        help="Path to the fine-tuned model adapter directory")
    parser.add_argument("--max_tokens", type=int, default=500,
                        help="Maximum tokens to generate in each response")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for sampling (higher = more random)")
    
    args = parser.parse_args()
    
    system_prompt = """I am an AI assistant fine-tuned on the Mahabharata, one of the world's greatest epics. I can answer questions about its characters, events, philosophical teachings, and cultural significance. Feel free to ask me anything about this ancient Sanskrit text!"""
    
    interactive_chat(
        args.model, 
        args.adapter_path, 
        system_prompt=system_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )

if __name__ == "__main__":
    main() 