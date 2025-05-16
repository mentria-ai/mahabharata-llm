#!/bin/bash
# Test script to run a small end-to-end test of the Mahabharata QA pipeline

set -e  # Exit on any error

echo "=== Mahabharata QA Pipeline Test ==="
echo "This script will run a small end-to-end test of the pipeline using a sample text."

# Create test directories if they don't exist
mkdir -p data/raw
mkdir -p data/chunks
mkdir -p data/outputs
mkdir -p data/final
mkdir -p logs

# Step 1: Generate sample text
echo "Step 1: Generating sample text..."
python scripts/sample_test.py --size 25000 --output data/raw/test_mahabharata.txt

# Step 2: Chunk the text
echo "Step 2: Chunking the text..."
python scripts/chunk_data.py --input data/raw/test_mahabharata.txt --output_dir data/chunks --chunk_size 5000

# Count how many chunks were created
CHUNK_COUNT=$(ls -1 data/chunks/chunk_*.txt | wc -l | tr -d ' ')
echo "Created $CHUNK_COUNT chunks."

# Step 3: Generate Q&A pairs for the first chunk only (to save time)
echo "Step 3: Generating Q&A pairs (for first chunk only)..."
python scripts/generate_xml.py --single_chunk "chunk_0001.txt" --chunks_dir data/chunks --output_dir data/outputs

# Step 4: Parse and format the output
echo "Step 4: Parsing and formatting output..."
python scripts/parse_and_format.py --input_dir data/outputs --output_file data/final/test_qa.jsonl

# Check if the output file was created
if [ -f data/final/test_qa.jsonl ]; then
    QA_COUNT=$(wc -l < data/final/test_qa.jsonl)
    echo "Success! Generated $QA_COUNT Q&A pairs."
    echo "Output file: data/final/test_qa.jsonl"
    
    # Print first few lines
    echo "Sample output:"
    head -n 3 data/final/test_qa.jsonl
else
    echo "Error: Failed to generate output file."
    exit 1
fi

echo "=== Test completed successfully ==="
echo "To run the full pipeline on your actual data:"
echo "1. Place your Mahabharata text in data/raw/"
echo "2. Run the pipeline scripts as documented in README.md" 