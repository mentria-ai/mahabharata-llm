# Mahabharata Q&A Generation Pipeline

A modular pipeline to generate fine-tuning data from a Sanskrit *Mahabharata* text using the Qwen3-30B-A3B model. The pipeline runs locally via MLX on Apple Silicon hardware, ensuring data privacy and leveraging local hardware acceleration.

## Pipeline Overview

The pipeline consists of several stages:

1. **Data Chunking**: Split the large input text into manageable chunks of ~10,000 characters each.
2. **Q&A Generation**: Use Qwen3-30B-A3B to generate question-answer pairs with reasoning for each chunk.
3. **Cleaning and Formatting**: Parse the model's output, remove reasoning, and format into a clean JSONL dataset.

## Directory Structure

```
MahabharataPipeline/
├── data/
│   ├── raw/        # Original input texts
│   ├── chunks/     # Chunked text files
│   ├── outputs/    # Model outputs with reasoning
│   └── final/      # Cleaned JSONL dataset
├── scripts/        # Pipeline scripts
├── checkpoints/    # Progress tracking
└── logs/           # Log files
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3)
- Python 3.8+
- MLX and mlx-lm packages
- Qwen3-30B-A3B-4bit model (MLX version)

## Installation

1. Install required Python packages:

```bash
pip install mlx mlx-lm
```

2. Make sure you have the Qwen3 model available or it will be downloaded automatically when first used.

## Usage

### 1. Chunking the Data

Place your Mahabharata text file in `data/raw/` and run:

```bash
cd MahabharataPipeline
python scripts/chunk_data.py --input data/raw/mahabharata.txt --output_dir data/chunks --chunk_size 10000
```

This will split the text into chunks of approximately 10,000 characters each, stored in `data/chunks/`.

### 2. Generating Q&A Pairs

Run the generation script to process all chunks:

```bash
python scripts/generate_xml.py --chunks_dir data/chunks --output_dir data/outputs
```

This will generate XML files with Q&A pairs and reasoning for each chunk. The process can be stopped and resumed at any time.

For parallel processing (faster but more memory-intensive):

```bash
python scripts/parallel_generate.py --num_processes 4
```

Adjust the number of processes based on your hardware capabilities. Be cautious with memory usage when running multiple Qwen3 instances simultaneously.

### 3. Cleaning and Creating the Dataset

Parse the generated outputs and create a clean JSONL dataset:

```bash
python scripts/parse_and_format.py --input_dir data/outputs --output_file data/final/mahabharata_qa.jsonl
```

This will remove the reasoning tags and format the data into a clean JSONL file ready for fine-tuning.

## Advanced Options

### Processing Specific Chunks

To process only a specific range of chunks:

```bash
python scripts/parallel_generate.py --chunk_range 1-100
```

### Adjusting Generation Parameters

You can modify generation parameters like temperature, top_p, and the number of Q&A pairs:

```bash
python scripts/generate_xml.py --temperature 0.8 --top_p 0.95 --num_qa_pairs 3
```

### Checkpointing and Resume

The pipeline automatically implements checkpointing - it saves after each chunk and can resume from where it left off if interrupted. This makes it resilient to crashes or stops.

## Output Format

The final JSONL file will contain entries like:

```json
{"question": "What was Arjuna's role in the Kurukshetra war?", "answer": "Arjuna was the third Pandava brother and the greatest archer in the Kurukshetra war. He was Krishna's charioteer and received the Bhagavad Gita from him before the battle."}
```

## Notes on Performance

- Qwen3-30B-A3B runs at approximately 30 tokens/sec on Apple Silicon
- Consider memory usage when running in parallel - each model instance uses significant RAM
- For the fastest results with quality trade-offs, consider using a smaller model like Qwen3-7B 

## Generating High-Quality Q&A Dataset

To generate a rich, contextually-aware, standalone dataset from the Mahabharata text:

```bash
# Generate 5 high-quality Q&A pairs per chunk with the optimized prompts
python scripts/generate_xml.py --chunks_dir data/chunks --output_dir data/outputs \
  --model_name "mlx-community/Qwen3-30B-A3B-4bit" \
  --num_qa_pairs 5 \
  --max_tokens 8000

# After generation completes, parse and format the outputs into a JSONL dataset
python scripts/parse_and_format.py --input_dir data/outputs --output_file data/final/mahabharata_qa.jsonl
```

The optimized prompts result in Q&A pairs that:
- Are completely standalone (never refer to "the passage" or "the text")
- Include necessary context for lesser-known characters, places, and concepts
- Present information in engaging formats (factual, "did you know," detective-style)
- Contain precise details from the Sanskrit text
- Balance accessibility with depth and accuracy

### Expected Processing Time

With the enhanced prompts and 5 Q&A pairs per chunk:
- Each chunk takes approximately 60-90 seconds to process
- The full 719 chunks will take roughly 12-18 hours to complete

You can safely interrupt the process at any time with Ctrl+C, and resume by running the same command later - it will skip already processed chunks.

### Monitoring Progress

Check how many chunks have been processed:
```bash
ls data/outputs/*.xml | wc -l
```

Compare to total chunks:
```bash
ls data/chunks/*.txt | wc -l
``` 