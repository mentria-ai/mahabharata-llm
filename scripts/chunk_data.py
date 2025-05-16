import os
import argparse


def _write_chunk(buffer: str, output_dir: str, chunk_num: int, unit_str: str):
    """Utility: persist the current buffer and report progress."""
    if not buffer:
        return
    chunk_path = os.path.join(output_dir, f"chunk_{chunk_num:04d}.txt")
    with open(chunk_path, "w", encoding="utf-8") as outfile:
        outfile.write(buffer)
    print(f"Wrote chunk {chunk_num} ({len(buffer)} {unit_str})")


def chunk_by_chars(input_path: str, output_dir: str, max_chars: int):
    """Classic splitter: limit by *character* count (default behaviour)."""
    os.makedirs(output_dir, exist_ok=True)
    chunk_num, buffer = 1, ""

    with open(input_path, "r", encoding="utf-8") as infile:
        for line in infile:
            if len(buffer) + len(line) > max_chars and buffer:
                _write_chunk(buffer, output_dir, chunk_num, "chars")
                chunk_num += 1
                buffer = ""
            buffer += line

    _write_chunk(buffer, output_dir, chunk_num, "chars")


def chunk_by_tokens(input_path: str, output_dir: str, max_tokens: int, tokenizer_name: str):
    """Token-aware splitter that uses the model's tokenizer to count tokens.

    This provides much tighter control when working close to the model's
    context window (e.g. 30-40K tokens for Qwen3-30B-A3B).
    """
    from transformers.models.auto.tokenization_auto import AutoTokenizer  # type: ignore

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    eos = "\n"  # we will join lines with newlines to preserve stanza breaks

    os.makedirs(output_dir, exist_ok=True)
    chunk_num, buffer_lines = 1, []

    def _current_token_len(lines):
        return len(tokenizer(eos.join(lines)).input_ids)

    with open(input_path, "r", encoding="utf-8") as infile:
        for raw_line in infile:
            line = raw_line.rstrip("\n")

            prospective_lines = buffer_lines + [line]
            if _current_token_len(prospective_lines) > max_tokens and buffer_lines:
                # flush current chunk
                _write_chunk(eos.join(buffer_lines), output_dir, chunk_num, "tokens")
                chunk_num += 1
                buffer_lines = []

            buffer_lines.append(line)

    if buffer_lines:
        _write_chunk(eos.join(buffer_lines), output_dir, chunk_num, "tokens")


def main():
    parser = argparse.ArgumentParser(description="Split a large text file into smaller chunks")
    parser.add_argument("--input", "-i", required=True, help="Input text file path")
    parser.add_argument("--output_dir", "-o", default="../data/chunks", help="Output directory for chunks")

    # Mutually exclusive char- vs token-based splitting
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--chunk_size", "-s", type=int, help="Target chunk size IN CHARACTERS")
    group.add_argument("--max_tokens", "-k", type=int, help="Target chunk size IN TOKENS (requires tokenizer)")

    parser.add_argument("--model_name", "-m", default="mlx-community/Qwen3-30B-A3B-4bit", help="Tokenizer name when --max_tokens is used")

    args = parser.parse_args()

    if args.max_tokens:
        chunk_by_tokens(args.input, args.output_dir, args.max_tokens, args.model_name)
    else:
        # Fallback to char splitter (default 10K if not provided)
        chunk_by_chars(args.input, args.output_dir, args.chunk_size or 10000)


if __name__ == "__main__":
    main() 