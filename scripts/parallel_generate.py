import os
import argparse
import json
import time
import multiprocessing
from functools import partial
import subprocess

def process_chunk(chunk_file, args):
    """Process a single chunk file using the generate_xml.py script"""
    chunk_path = os.path.join(args.chunks_dir, chunk_file)
    chunk_id = chunk_file.split('.')[0]  # e.g., "chunk_0001"
    output_path = os.path.join(args.output_dir, f"{chunk_id}_out.xml")
    
    # Skip if output already exists
    if os.path.exists(output_path) and not args.force:
        print(f"Output for {chunk_file} already exists, skipping.")
        return {"chunk": chunk_file, "status": "skipped", "time": 0}
    
    start_time = time.time()
    
    # Build command to run generate_xml.py for a single chunk
    cmd = [
        "python", 
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_xml.py"),
        "--model_name", args.model_name,
        "--chunks_dir", args.chunks_dir,
        "--output_dir", args.output_dir,
        "--num_qa_pairs", str(args.num_qa_pairs),
        "--max_tokens", str(args.max_tokens),
        "--temperature", str(args.temperature),
        "--top_p", str(args.top_p),
        "--single_chunk", chunk_file  # We'll add this parameter to generate_xml.py
    ]
    
    try:
        # Run the command and capture output
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        elapsed = time.time() - start_time
        return {
            "chunk": chunk_file, 
            "status": "success", 
            "time": elapsed,
            "stdout": result.stdout
        }
    except subprocess.CalledProcessError as e:
        print(f"Error processing {chunk_file}: {e}")
        return {
            "chunk": chunk_file, 
            "status": "error",
            "error": str(e),
            "stdout": e.stdout,
            "stderr": e.stderr
        }

def main():
    parser = argparse.ArgumentParser(description='Process chunks in parallel with multiple processes')
    parser.add_argument('--model_name', '-m', default="mlx-community/Qwen3-30B-A3B-4bit", 
                      help='MLX model repository name')
    parser.add_argument('--chunks_dir', '-c', default="../data/chunks", 
                      help='Directory containing text chunks')
    parser.add_argument('--output_dir', '-o', default="../data/outputs", 
                      help='Directory to save model outputs')
    parser.add_argument('--log_file', '-l', default="../logs/parallel_generate.json", 
                      help='JSON log file for processing results')
    parser.add_argument('--num_processes', '-p', type=int, default=None, 
                      help='Number of parallel processes (default: CPU count)')
    parser.add_argument('--chunk_range', '-r', type=str, default=None, 
                      help='Range of chunks to process (format: start-end, e.g., 1-100)')
    parser.add_argument('--force', '-f', action='store_true', 
                      help='Force reprocessing of chunks that already have outputs')
    parser.add_argument('--num_qa_pairs', '-n', type=int, default=2, 
                      help='Number of QA pairs to generate per chunk')
    parser.add_argument('--max_tokens', type=int, default=1024, 
                      help='Maximum tokens to generate per response')
    parser.add_argument('--temperature', '-t', type=float, default=0.7, 
                      help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9, 
                      help='Nucleus sampling probability')
    
    args = parser.parse_args()
    
    # Create output and log directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    
    # Get list of chunk files
    all_chunks = sorted([f for f in os.listdir(args.chunks_dir) if f.endswith('.txt')])
    
    # Filter chunks by range if specified
    if args.chunk_range:
        try:
            start, end = map(int, args.chunk_range.split('-'))
            filtered_chunks = []
            for chunk in all_chunks:
                # Extract chunk number (assuming format like chunk_0001.txt)
                try:
                    chunk_num = int(chunk.split('_')[1].split('.')[0])
                    if start <= chunk_num <= end:
                        filtered_chunks.append(chunk)
                except (ValueError, IndexError) as e:
                    print(f"Warning: Could not parse chunk number from {chunk}: {e}")
                    continue
            all_chunks = filtered_chunks
            print(f"Filtered to {len(all_chunks)} chunks in range {start}-{end}")
        except (ValueError, IndexError) as e:
            print(f"Invalid chunk range format: {args.chunk_range}. Using all chunks. Error: {e}")
    
    total_chunks = len(all_chunks)
    print(f"Processing {total_chunks} chunks using {args.num_processes or multiprocessing.cpu_count()} processes")
    
    # Process chunks in parallel
    with multiprocessing.Pool(processes=args.num_processes) as pool:
        process_func = partial(process_chunk, args=args)
        results = list(pool.map(process_func, all_chunks))
    
    # Count successful and failed chunks
    successes = sum(1 for r in results if r['status'] == 'success')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    errors = sum(1 for r in results if r['status'] == 'error')
    
    # Save results to log file
    with open(args.log_file, 'w', encoding='utf-8') as f:
        json.dump({
            "total": total_chunks,
            "successes": successes,
            "skipped": skipped,
            "errors": errors,
            "results": results
        }, f, indent=2)
    
    print(f"Processing complete: {successes} successful, {skipped} skipped, {errors} errors")
    print(f"Log saved to: {args.log_file}")

if __name__ == "__main__":
    main() 