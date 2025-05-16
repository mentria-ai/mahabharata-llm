# Fine-Tuning Qwen3-30B-A3B for Mahabharata Q&A

This directory contains scripts and configurations for fine-tuning the Qwen3-30B-A3B model using MLX on Apple Silicon. The fine-tuning uses QLoRA (Quantized Low-Rank Adaptation) to efficiently train the model on a custom Mahabharata Q&A dataset while minimizing memory usage.

## Directory Structure

```
finetune/
├── data/                   # Training and validation data
│   ├── train.jsonl         # Training dataset (prompt/completion format)
│   └── valid.jsonl         # Validation dataset
├── adapters/               # Fine-tuned model adapters (LoRA weights)
├── finetune_config.yaml    # Configuration for fine-tuning
├── test_questions.jsonl    # Sample questions for evaluation
└── evaluation_results.csv  # Evaluation results (base vs fine-tuned)
```

## Prerequisites

- macOS with Apple Silicon (Mac mini M4 Pro or equivalent)
- Python 3.8+ with MLX and MLX-LM installed
- Qwen3-30B-A3B-4bit model (MLX version)

## Fine-Tuning Pipeline

The fine-tuning process consists of three main steps:

1. **Data Preparation**: Convert Q&A dataset to prompt/completion format
2. **Model Fine-Tuning**: Use QLoRA to fine-tune Qwen3 on the dataset
3. **Evaluation**: Compare the base model vs. fine-tuned model

### 1. Data Preparation

Convert your Q&A dataset to the prompt/completion format required by MLX:

```bash
cd MahabharataPipeline
python scripts/prepare_finetune_data.py --input data/final/mahabharata_qa.jsonl --output_dir finetune/data --val_split 0.1
```

This script:
- Reads the input JSONL with "question" and "answer" fields
- Converts to format with "prompt" and "completion" fields
- Splits data into training (90%) and validation (10%) sets
- Saves as train.jsonl and valid.jsonl in finetune/data/

### 2. Fine-Tuning

Run the fine-tuning process:

```bash
python scripts/finetune.py --config finetune/finetune_config.yaml
```

This script:
- Prepares the dataset if not already done
- Runs MLX-LM's LoRA fine-tuning using the configuration file
- Outputs training logs and saves checkpoints in finetune/adapters/

The process uses QLoRA with these optimizations:
- 4-bit quantization of the base model
- Training only the top 8 layers (memory efficient)
- Gradient checkpointing for reduced memory usage
- Small batch size (1) to fit on 24GB memory

Fine-tuning will take several hours, depending on iteration count.

### 3. Evaluation

After fine-tuning, evaluate the model against the base model:

```bash
python scripts/evaluate_model.py --adapter_path finetune/adapters
```

This will:
- Sample 5 test questions from the dataset
- Generate responses using both the base and fine-tuned models
- Compare the responses and timing
- Save results to finetune/evaluation_results.csv

### Interactive Chat

Test the fine-tuned model interactively:

```bash
python scripts/chat.py --adapter_path finetune/adapters
```

This launches an interactive chat interface where you can ask questions about the Mahabharata.

## Configuration Options

The `finetune_config.yaml` file contains all the parameters for fine-tuning:

- `model`: The model to fine-tune (using 4-bit quantized version)
- `batch_size`: Default is 1 for memory efficiency (increase if you have more memory)
- `iters`: Number of training iterations (default: 1000)
- `learning_rate`: Learning rate for fine-tuning (default: 2e-4)
- `num_layers`: Number of layers to fine-tune (default: 8)

You can modify these parameters to adjust the fine-tuning process.

## Tips for Successful Fine-Tuning

1. **Memory Management**: If you encounter out-of-memory errors:
   - Reduce batch size to 1
   - Reduce num_layers to 4 or fewer
   - Enable gradient checkpointing
   - Ensure you're using the 4-bit quantized model

2. **Training Duration**: 
   - For the ~7000 Q&A pairs, ~1000-2000 iterations is a good starting point
   - Monitor training loss and validation metrics to avoid overfitting

3. **Evaluation**:
   - Compare generated answers for factual accuracy
   - Check if the fine-tuned model adopts the training data's style
   - Look for improvements in relevance and specificity

## Advanced: Fusing the Model

After fine-tuning, you can optionally fuse the LoRA adapter with the base model:

```bash
mlx_lm.fuse --model mlx-community/Qwen3-30B-A3B-4bit --adapter-path finetune/adapters --output-path finetune/fused_model --trust-remote-code
```

This creates a standalone model with the adapter weights integrated. 