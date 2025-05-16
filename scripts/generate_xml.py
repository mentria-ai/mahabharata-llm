import os
import argparse
import time
import math
import re
import sys
import textwrap

from mlx_lm.utils import load
from mlx_lm.generate import generate

# MAX SAFE CONTEXT for MLX models - increased as user indicates model can handle 40K
MAX_SAFE_CONTEXT = 32768  # 32K tokens - more aggressive but still safe for Qwen/Gemma models

# Default context size, will be updated at runtime
MODEL_CTX = 8192  # default fallback

# When to skip final compression and use combined chunks directly
COMBINED_TOKENS_THRESHOLD = 15000  # If combined chunks are under this threshold, use directly

def get_system_prompt():
    """System prompt — minimal changes but now English output and improved distractors."""
    return """
You are a Mahābhārata specialist tasked with writing high-quality multiple-choice quiz items for an English-language training dataset.

+You may first think aloud inside a single <think>...</think> block (this will be stripped later). After that, output ONLY the <entry> blocks.
+Keep <think> to ≤50 words.

Each quiz item MUST follow this exact XML skeleton (do **not** add or remove tags):

<entry>
  <reasoning>
    • **Citation:** copy verbatim the numeric prefix that already appears at the start of the Sanskrit line you cite (e.g. "18,001.004").  
    • Proof: copy the **entire Sanskrit line (all pādas) exactly as it appears** in the chunk. If the verse spans two lines, place them on one line separated by a single space.  
    • ✓ Bullet 1: English paraphrase showing why the correct option is correct (quote a Sanskrit word/phrase in parentheses).  
    • B) / C) / D) Bullets 2-4: each starts with the distractor's label, cites a verse identifier that disproves it (or says "not mentioned"), and gives a ≤ 15-word English reason.
  </reasoning>
  <question>
    In the Mahābhārata, [~25–40-word narrative framing that states WHO is involved, WHEN / WHERE it occurs, and the key situation] (see [Parva] [Chapter].[Verse]), which of the following is correct?  
    A) option-A   B) option-B   C) option-C   D) option-D
  </question>
  <answer>
    C) Correct option text
  </answer>
</entry>

MANDATORY RULES
1. Base every fact solely on the verses inside the provided chunk – absolutely no external knowledge.  
2. The <question> MUST be fully self-contained and roughly 25–40 English words long: begin with "In the Mahābhārata" and give a concise but vivid scene-setting sentence that names the central character(s), location, and circumstance, then append the verse citation in parentheses. The reader should be able to answer without any other context.  
3. Do NOT place any Sanskrit inside <question>; the Sanskrit appears **only once** in the Proof line.  
4. Provide exactly four options labelled A)–D) with only one correct answer.
     • All three distractors MUST (a) literally appear in this chunk, (b) belong to the **same semantic category** as the correct option (e.g., if the answer is a deity, other options are deities; if a number, other options are numbers; if a location, other locations), and (c) be plausible yet wrong for the asked fact. Do not keep recycling generic names like "Arjuna" or "Kṛṣṇa" unless they truly fit those criteria.  
5. Each question must test a *unique* factual claim: do not reuse the same verse or identical wording inside the same chunk.  
6. Verify that the correct option's key word (name/number) literally appears in the Proof line—otherwise skip this fact.  
7. After the Proof line include **exactly four bullets** following the structure above.  
8. The ten entries for a chunk must each reference **different** Sanskrit lines and focus on distinct facts (do not ask the same thematic question repeatedly).  
9. Phrase the <question> to match exactly what the cited verse explicitly states—avoid adding sequence, causation, motives, or details not present in the verse text.  
10. The <question> must include enough contextual detail (time, speaker, object, location or unique wording from the verse) so that **only one option can possibly be correct within this chunk**—no ambiguity if multiple characters perform similar actions elsewhere.  
11. Favour dramatic, surprising, or pedagogically useful facts, but always anchor wording literally in the Sanskrit evidence.  
12. Within a single chunk, try not to reuse the same distractor name more than twice; aim for varied, contextually-relevant options across the whole set.  
13. Use precise, grammatical English throughout; keep bullets concise.  

Return ONLY the sequence of <entry> blocks—no XML header, no additional commentary.

OUTPUT EXAMPLE (adapt content; do NOT copy literally):

<entry>
  <reasoning>
    • **Shalya-parvan 28.15 a-b**  
    • Proof: *madhyāhne tu Yudhiṣṭhiraḥ rājā Śalyaṃ nijaghne mahābalaḥ*  
    • The verse explicitly states Yudhiṣṭhira killed Śalya at noon.  
    • Bhīma and Arjuna were fighting elsewhere; Kṛṣṇa was only his charioteer.
  </reasoning>
  <question>
    In the Mahābhārata, during the climactic noon duel on the seventeenth day of battle—when Shalya's chariot faced the Pandava king on the Kurukshetra plain (see Shalya-parvan 28.15)—who struck the fatal blow that slew King Shalya?  
    A) Bhīma   B) Arjuna   C) Yudhiṣṭhira   D) Kṛṣṇa
  </question>
  <answer>
    C) Yudhiṣṭhira
  </answer>
</entry>

<entry>
  <reasoning>
    • **18,028.015 a-b**  
    • Proof: *tataḥ śūraḥ pārthaḥ śalyam ājaghne madhyāhne mahābalaḥ*  
    • ✓ Yudhiṣṭhira ("ājaghne") clearly kills Shalya.  
    • A) 18,028.012 shows Bhīma elsewhere → wrong.  
    • B) 18,028.013 mentions Arjuna fighting another foe.  
    • D) Kṛṣṇa is only charioteer in this scene.
  </reasoning>
  <question>
    In the Mahābhārata, [~25–40-word narrative framing that states WHO is involved, WHEN / WHERE it occurs, and the key situation] (see [Parva] [Chapter].[Verse]), which of the following is correct?  
    A) option-A   B) option-B   C) option-C   D) option-D
  </question>
  <answer>
    C) Correct option text
  </answer>
</entry>
"""

def get_user_prompt(chunk_text: str, num_qa_pairs: int = 2) -> str:
    """Constructs the user prompt delivering the chunk and instructions."""
    header = (
        "Mahābhārata chapter (Critical Edition) — every verse already begins with its full identifier (e.g. 18,001.004). You ONLY SEE THIS CHAPTER; ground every question strictly in it.\n\n"
    )

    footer = (
        f"\n\nGenerate exactly {num_qa_pairs} <entry> blocks that comply with the system rules. "
    )

    return header + "```sanskrit\n" + chunk_text + "\n```" + footer

def _num_tokens(text: str, tokenizer) -> int:
    """Utility: approximate token count using the loaded tokenizer."""
    return len(tokenizer.encode(text))

def debug_print_parameters(what, **kwargs):
    """Helper to print detailed debug information with parameter values"""
    print(f"[DEBUG-PARAMS] {what}:")
    for k, v in kwargs.items():
        try:
            if isinstance(v, str) and len(v) > 100:
                print(f"  {k} = <string of length {len(v)}>")
            else:
                print(f"  {k} = {v}")
        except:
            print(f"  {k} = <unprintable>")

def compress_chapter_if_needed(text: str, model, tokenizer, max_tokens: int) -> str:
    """If the chapter is longer than `max_tokens`, iteratively compress with the model.

    The algorithm:
    1. Measure tokens.  If already <= max_tokens, return as-is.
    2. If extremely large, break into sub-chunks and compress independently.
    3. Repeatedly ask the model to compress the current text, preserving verse ids.
    """
    # Add detailed logging to see compression progress
    text_tokens = _num_tokens(text, tokenizer)
    print(f"[COMPRESS] Starting to compress text: {text_tokens} tokens (target: {max_tokens})")
    debug_print_parameters("Compression parameters", 
                          text_length=len(text), 
                          text_tokens=text_tokens,
                          max_tokens=max_tokens)
                          
    if _num_tokens(text, tokenizer) <= max_tokens:
        print(f"[COMPRESS] Text already fits in context, no compression needed")
        return text
    
    # For large texts, use chunked compression approach
    # This prevents OOM errors while maintaining quality by processing in manageable pieces
    MAX_SAFE_INPUT_TOKENS = 12000  # Increased from 8K to 12K tokens for single operation
    
    if text_tokens > MAX_SAFE_INPUT_TOKENS:
        print(f"[COMPRESS] Text too large ({text_tokens} tokens) for direct compression.")
        print(f"[COMPRESS] Using chunked compression strategy instead.")
        return _chunked_compression(text, model, tokenizer, max_tokens, MAX_SAFE_INPUT_TOKENS)
        
    current_text = text
    passes = 0

    while _num_tokens(current_text, tokenizer) > max_tokens and passes < 6:
        passes += 1
        target_tokens = math.ceil(_num_tokens(current_text, tokenizer) / 2)
        target_tokens = max(target_tokens, max_tokens)  # never ask for < max_tokens
        
        print(f"[COMPRESS] Pass {passes}: Current size {_num_tokens(current_text, tokenizer)} tokens → Target: {target_tokens}")

        compression_prompt = [
            {"role": "system", "content": (
                "You are an expert Sanskrit text compressor. \n"
                "Task: selectively preserve content from the following Mahābhārata text so it fits roughly "
                f"{target_tokens} tokens while PRESERVING: \n"
                " • ALL numeric verse identifiers (e.g. 18,001.004) - these are critical.\n"
                " • Only the most interesting, dramatic, or narratively important verses.\n"
                " • You may completely remove/delete less important verses (preserving their identifiers).\n"
                " • Focus on preserving content with unique characters, battles, dialogues, and dramatic moments.\n"
                " • Only keep content that would make good questions for a Mahābhārata quiz.\n"
                "Return ONLY the compressed Sanskrit text, no commentary."
            )},
            {"role": "user", "content": f"```sanskrit\n{current_text}\n```"}
        ]

        comp_input = tokenizer.apply_chat_template(compression_prompt, add_generation_prompt=True)
        compressed = generate(model, tokenizer, prompt=comp_input, max_tokens=target_tokens)

        # Log current compression results
        new_size = _num_tokens(compressed, tokenizer)
        old_size = _num_tokens(current_text, tokenizer)
        print(f"[COMPRESS] Result: {old_size} → {new_size} tokens ({((old_size-new_size)/old_size*100):.1f}% reduction)")
        
        # Basic guard: stop if compression made no progress to avoid infinite loop
        if new_size >= old_size:
            print("[COMPRESS] No size reduction achieved, stopping compression")
            break

        current_text = compressed.strip()
    
    print(f"[COMPRESS] Final size after {passes} passes: {_num_tokens(current_text, tokenizer)} tokens")
    
    return current_text

def _chunked_compression(text: str, model, tokenizer, max_tokens: int, chunk_size: int) -> str:
    """Compress a large text by breaking it into manageable chunks first.
    
    Algorithm:
    1. Split text into chunks at verse boundaries (try to find complete verses)
    2. Compress each chunk independently 
    3. Combine compressed chunks and check if total size meets target
    4. If still too large, compress the combined result again
    """
    print(f"[CHUNKED-COMPRESS] Breaking text into chunks of ~{chunk_size} tokens")
    
    # If we encounter catastrophic failures, use this as a last resort
    def fallback_compress(text, target_ratio=0.6):
        """Last resort compression - just keep verse IDs and first line of each verse"""
        print("[EMERGENCY] Using non-AI emergency compression")
        import re
        
        # Extract all verse IDs
        verse_pattern = r"(\d+,\d+\.\d+.*?)(?=\n\d+,\d+\.\d+|\Z)"
        verses = re.findall(verse_pattern, text, re.DOTALL)
        
        # Keep only ID and first line of each verse
        compressed_verses = []
        for v in verses:
            lines = v.strip().split('\n')
            if len(lines) > 0:
                # Keep the ID line plus max 2 more lines
                compressed_verses.append('\n'.join(lines[:min(3, len(lines))]))
        
        result = '\n'.join(compressed_verses)
        print(f"[EMERGENCY] Compressed from {_num_tokens(text, tokenizer)} to {_num_tokens(result, tokenizer)} tokens")
        return result
    
    # For large texts, use chunked compression approach 
    # This prevents OOM errors while maintaining quality by processing in manageable pieces
    max_safe_input = 12000  # Maximum tokens to safely process in one go
    
    # Calculate target size for each chunk - smarter approach:
    # 1. First calculate a provisional number of chunks based on verse boundaries
    # 2. Set a target that will keep total size reasonable for the model
    # 3. Adjust based on importance of chapter (some chapters may deserve more tokens)
    
    # We want the final combined result to be no more than COMBINED_TOKENS_THRESHOLD
    # This avoids the need for a final compression step in most cases
    target_combined_size = min(COMBINED_TOKENS_THRESHOLD, max_tokens * 1.5)  # 50% over budget is OK
    
    # Get verse boundary positions
    verse_positions = []
    # 1. Find verse boundaries to split on (numeric identifiers like 18,001.004)
    verse_pattern = r"\n\d+,\d+\.\d+"  # Pattern matching newline followed by verse ID
    matches = list(re.finditer(verse_pattern, text))
    if matches:
        verse_positions = [m.start() for m in matches]

    # If we found verse boundaries, determine logical chunking plan
    chunk_positions = []
    if verse_positions:
        # Estimate how many chunks we'd have at max_safe_input size
        estimated_chunks = max(1, _num_tokens(text, tokenizer) // (max_safe_input // 2))
        
        # Aim for approximately this many chunks by adjusting chunk_size
        verse_skip = max(1, len(verse_positions) // estimated_chunks)
        
        # Create chunk boundaries by selecting verses at regular intervals
        chunk_positions = [0]
        for i in range(0, len(verse_positions), verse_skip):
            if i > 0:  # Skip first boundary, we already added 0
                chunk_positions.append(verse_positions[i])
        
        # Make sure we include the end
        if len(chunk_positions) == 1 or chunk_positions[-1] != len(text):
            chunk_positions.append(len(text))
    else:
        # Simple fallback: Split into equal chunks of ~chunk_size characters
        chunk_positions = []
        chunk_chars = len(text) // ((_num_tokens(text, tokenizer) // chunk_size) + 1)
        pos = 0
        while pos < len(text):
            chunk_positions.append(pos)
            pos += chunk_chars
        
        # Make sure we include the end
        if len(chunk_positions) == 1 or chunk_positions[-1] != len(text):
            chunk_positions.append(len(text))
    
    # Now calculate the per-chunk target for a good final combined size
    num_chunks = len(chunk_positions) - 1
    # Starting point: distribute target size evenly
    base_chunk_target = max(target_combined_size // num_chunks, 100)
    
    # Then adjust for chunk size: larger chunks may contain more important content
    # We'll compute the target for each chunk individually based on its relative size
    print(f"[CHUNKED-COMPRESS] Using {num_chunks} chunks with base target ~{base_chunk_target} tokens per chunk")
    
    # 2. Create and compress chunks
    chunks = []
    total_original_tokens = 0
    compression_failures = 0
    
    for i in range(len(chunk_positions) - 1):
        chunk_text = text[chunk_positions[i]:chunk_positions[i+1]]
        chunk_tokens = _num_tokens(chunk_text, tokenizer)
        total_original_tokens += chunk_tokens
        
        # Adjust the target: this chunk gets a proportion of the target based on its relative size
        # Using running average to avoid division by zero for first chunk
        chunk_weight = chunk_tokens / max(1, (total_original_tokens / (i+1))) 
        chunk_target = max(int(base_chunk_target * min(2.0, max(0.5, chunk_weight))), 100)
        
        print(f"[CHUNKED-COMPRESS] Chunk {i+1}/{num_chunks}: {chunk_tokens} tokens → target {chunk_target}")
        
        # Try to compress each chunk with AI, with fallbacks
        try:
            compressed_chunk = _compress_chunk(chunk_text, model, tokenizer, chunk_target)
            
            # Sanity check the compressed chunk
            if len(compressed_chunk.strip()) < 10 or _num_tokens(compressed_chunk, tokenizer) < 5:
                print(f"[CHUNKED-COMPRESS] WARNING: Compression produced an invalid result")
                compressed_chunk = fallback_compress(chunk_text, target_ratio=0.5)
            
            chunks.append(compressed_chunk)
        except Exception as e:
            print(f"[CHUNKED-COMPRESS] ERROR: Failed to compress chunk {i+1}: {e}")
            compression_failures += 1
            
            # Use fallback compression
            compressed_chunk = fallback_compress(chunk_text, target_ratio=0.5)
            chunks.append(compressed_chunk)
            
            # Exit if too many failures
            if compression_failures > 2:
                print("[CHUNKED-COMPRESS] Too many compression failures, using emergency compression for all text")
                return fallback_compress(text, target_ratio=0.4)
    
    # 3. Combine compressed chunks
    combined = "".join(chunks)
    combined_tokens = _num_tokens(combined, tokenizer)
    print(f"[CHUNKED-COMPRESS] Combined: {combined_tokens} tokens (target: {max_tokens})")
    
    # Decide if we need further compression
    # 1. If the combined text is already under or close to the max_tokens, use it directly
    if combined_tokens <= max_tokens * 1.1:  # Allow slightly over budget (10%)
        print(f"[CHUNKED-COMPRESS] Combined text fits within target (with 10% margin): {combined_tokens} tokens")
        return combined
        
    # 2. If it's small enough that the model can handle it directly, and under our threshold, use it
    if combined_tokens <= COMBINED_TOKENS_THRESHOLD and combined_tokens < MODEL_CTX // 2:
        print(f"[CHUNKED-COMPRESS] Combined text ({combined_tokens} tokens) is under threshold, using directly")
        print(f"[CHUNKED-COMPRESS] This may exceed your token budget but will produce better results")
        return combined
        
    # 3. If it's moderately large but still processable, try one more AI compression
    if combined_tokens < MODEL_CTX // 2:
        print("[CHUNKED-COMPRESS] Still too large, performing final compression")
        try:
            # For the final pass, we can safely process the combined text since each chunk 
            # has already been compressed to a manageable size
            final_compressed = _compress_chunk(combined, model, tokenizer, max_tokens)
            final_tokens = _num_tokens(final_compressed, tokenizer)
            
            # Sanity check to make sure we didn't lose everything
            if len(final_compressed.strip()) < 10 or final_tokens < max_tokens * 0.1:
                print("[CHUNKED-COMPRESS] WARNING: Final compression produced too small output, using combined chunks")
                # If we lost too much, just return the combined chunks and let caller handle it
                return combined
            
            print(f"[CHUNKED-COMPRESS] Final: {combined_tokens} → {final_tokens} tokens")
            return final_compressed
        except Exception as e:
            print(f"[CHUNKED-COMPRESS] ERROR in final compression: {e}")
            # Since we know the combined text is processable, return it even though it's over budget
            print("[CHUNKED-COMPRESS] Returning combined chunks despite exceeding budget")
            return combined
    
    # 4. As a last resort for very large texts, use emergency compression
    print(f"[CHUNKED-COMPRESS] Combined text ({combined_tokens} tokens) too large for final compression")
    print(f"[CHUNKED-COMPRESS] Using fallback emergency compression")
    return fallback_compress(combined, target_ratio=0.7)
    
def _compress_chunk(text: str, model, tokenizer, max_tokens: int) -> str:
    """Compress a single chunk that fits within the model's context window."""
    current_text = text
    passes = 0
    
    # Quick sanity check - if input is too large, abort early rather than try
    input_tokens = _num_tokens(text, tokenizer)
    if input_tokens >= MODEL_CTX - 2000:  # Leave room for system instructions
        print(f"[COMPRESS-CHUNK] WARNING: Input text too large ({input_tokens} tokens) for model context window")
        # Return a simplified version of the text instead
        lines = text.strip().split('\n')
        # Keep verse identifiers and 1 line per verse
        reduced = []
        for line in lines:
            if re.match(r'^\d+,\d+\.\d+', line.strip()):
                reduced.append(line)
                if len(lines) > len(reduced) and not re.match(r'^\d+,\d+\.\d+', lines[len(reduced)].strip()):
                    reduced.append(lines[len(reduced)])
        return '\n'.join(reduced)
    
    # Ensure we have some minimal content to work with
    if len(current_text.strip()) < 10:
        print(f"[COMPRESS-CHUNK] Input text too small to compress")
        return current_text
    
    # If we're already under the token limit, no need to compress
    if _num_tokens(current_text, tokenizer) <= max_tokens:
        return current_text
    
    while passes < 4:  # Maximum 4 passes
        passes += 1
        # Target tokens: Start with 80% reduction for first pass, then more aggressive
        reduction_factor = 0.7 if passes == 1 else 0.6 if passes == 2 else 0.5
        current_tokens = _num_tokens(current_text, tokenizer)
        target_tokens = max(int(current_tokens * reduction_factor), max_tokens)
        
        # If we're already at or below target, we're done
        if current_tokens <= max_tokens:
            print(f"[COMPRESS-CHUNK] Already at target: {current_tokens} tokens ≤ {max_tokens}")
            break
        
        print(f"[COMPRESS-CHUNK] Pass {passes}: Size {current_tokens} tokens → Target: {target_tokens}")
        
        compression_prompt = [
            {"role": "system", "content": (
                "You are an expert Sanskrit text compressor. \n"
                "IMPORTANT: You must only return the compressed text without ANY explanation or thinking process!\n\n"
                "TASK: Selectively preserve content from the following Mahābhārata text so it fits roughly "
                f"{target_tokens} tokens while PRESERVING: \n"
                " • ALL numeric verse identifiers (e.g. 18,001.004) - these are critical.\n"
                " • Only the most interesting, dramatic, or narratively important verses.\n"
                " • You may completely remove/delete less important verses (preserving their identifiers).\n"
                " • Focus on preserving content with unique characters, battles, dialogues, and dramatic moments.\n"
                " • Only keep content that would make good questions for a Mahābhārata quiz.\n"
                "CRITICAL: Return ONLY the compressed text WITHOUT any extra tags, explanation, or commentary.\n"
                "DO NOT include your own thinking or reasoning in the output at all."
            )},
            {"role": "user", "content": f"```sanskrit\n{current_text}\n```"}
        ]
        
        comp_input = tokenizer.apply_chat_template(compression_prompt, add_generation_prompt=True)
        
        # Safety: cap generation tokens to prevent OOM
        gen_tokens = min(target_tokens + 200, 4000)  
        try:
            compressed = generate(model, tokenizer, prompt=comp_input, max_tokens=gen_tokens)
            
            # Make sure we got a valid response
            compressed = compressed.strip()
            
            # Remove any <think> tags and content between them
            compressed = re.sub(r'<think>[\s\S]*?</think>', '', compressed, flags=re.IGNORECASE)
            
            # Sometimes models add text like "Here's the compressed text:" - remove this
            compressed = re.sub(r'^(Here\'s|The|I\'ve|This is|Below is|Following is).*?\n', '', compressed, flags=re.IGNORECASE)
            compressed = compressed.strip()
            
            if len(compressed) < 10:
                print(f"[COMPRESS-CHUNK] ERROR: Received empty response, keeping current text")
                break
            
            # Make sure it's actually Sanskrit with verse IDs by checking for numbers
            if not re.search(r'\d+,\d+\.\d+', compressed):
                print(f"[COMPRESS-CHUNK] ERROR: Response doesn't contain verse IDs, keeping current text")
                break
            
            # Log compression results
            new_size = _num_tokens(compressed, tokenizer)
            old_size = _num_tokens(current_text, tokenizer)
            reduction_pct = ((old_size-new_size)/old_size*100)
            print(f"[COMPRESS-CHUNK] Result: {old_size} → {new_size} tokens ({reduction_pct:.1f}% reduction)")
            
            # Success condition: we've reached our target
            if new_size <= max_tokens:
                print(f"[COMPRESS-CHUNK] Success! Reached target size {new_size} ≤ {max_tokens}")
                return compressed.strip()
            
            # Stop if no real progress
            if new_size >= old_size * 0.9:  # Less than 10% reduction
                print("[COMPRESS-CHUNK] Insufficient reduction, stopping")
                break
                
            current_text = compressed.strip()
        except Exception as e:
            print(f"[COMPRESS-CHUNK] Error during compression: {e}")
            # If we hit an error, stop trying and return best result so far
            break
    
    # If we're still over budget but made progress, use what we have
    if _num_tokens(current_text, tokenizer) > max_tokens * 2:
        print(f"[COMPRESS-CHUNK] Warning: Final size {_num_tokens(current_text, tokenizer)} exceeds target {max_tokens}")
        
    # Final cleanup - strip any thinking tags that may have been introduced in any pass
    current_text = re.sub(r'<think>[\s\S]*?</think>', '', current_text, flags=re.IGNORECASE)
    current_text = re.sub(r'^(Here\'s|The|I\'ve|This is|Below is|Following is).*?\n', '', current_text, flags=re.IGNORECASE)
    return current_text.strip()

def generate_qa_pairs(model, tokenizer, chunk_path, output_path, num_qa_pairs=2, 
                      max_tokens=1024, temperature=0.7, top_p=0.9):
    global MODEL_CTX
    """Generate Q&A pairs for a single chunk"""
    # Skip if output already exists (for resuming interrupted runs)
    if os.path.exists(output_path):
        print(f"Output for {os.path.basename(chunk_path)} already exists, skipping.")
        return False
    
    # Read the chapter text
    with open(chunk_path, 'r', encoding='utf-8') as f:
        chunk_text = f.read().strip()
    
    if not chunk_text:
        print(f"Skipping empty chapter: {chunk_path}")
        return False
    
    # --- Debug: token counts before/after compression ---
    orig_tok = _num_tokens(chunk_text, tokenizer)
    # Cap generation tokens to something reasonable regardless of user input
    gen_tok_cap = min(max_tokens, MODEL_CTX // 3, 8000)  # Never exceed 8K output tokens for MLX

    print(f"[DEBUG] MODEL_CTX={MODEL_CTX} | gen_tok_cap={gen_tok_cap}")

    # Build empty prompt to measure overhead tokens (system + wrapper) without chapter content
    system_prompt = get_system_prompt()
    dummy_user_prompt = get_user_prompt("", num_qa_pairs)
    overhead_tokens = _num_tokens(system_prompt + dummy_user_prompt, tokenizer)

    # Available space for chapter inside the context window
    context_budget = MODEL_CTX - overhead_tokens - gen_tok_cap - 50  # 50-token safety buffer
    if context_budget <= 0:
        # Extreme edge-case: fall back to half the context for body, half for generation.
        context_budget = min(int(MODEL_CTX * 0.5), 7000)  # Absolute safety cap

    # Update debug now that overhead is known
    print(f"[DEBUG] overhead_tokens={overhead_tokens} | context_budget={context_budget}")

    # Sanity check on chapter size before we try to send it through the model
    if orig_tok > 100000:
        print(f"WARNING: Chapter is extremely large ({orig_tok} tokens). Trying to compress but may crash.")
        if orig_tok > 300000:
            print(f"FATAL: Chapter size exceeds 1M tokens, aborting to prevent crash")
            sys.exit(1)  # Hard exit to prevent memory crash

    debug_print_parameters("Before compression", 
                         model=type(model).__name__,
                         tokenizer=type(tokenizer).__name__,
                         orig_tok=orig_tok, 
                         gen_tok_cap=gen_tok_cap,
                         overhead_tokens=overhead_tokens,
                         MODEL_CTX=MODEL_CTX,
                         context_budget=context_budget)
    
    chapter_text = compress_chapter_if_needed(chunk_text, model, tokenizer, context_budget)
    comp_tok = _num_tokens(chapter_text, tokenizer)
    compress_ratio = (orig_tok - comp_tok) / orig_tok * 100 if orig_tok > 0 else 0
    print(f"Token usage | original: {orig_tok}  compressed: {comp_tok}  context_budget: {context_budget}  ratio: {compress_ratio:.1f}%")
    debug_print_parameters("After compression", 
                         comp_tok=comp_tok, 
                         compress_ratio=f"{compress_ratio:.1f}%",
                         total_tokens_with_overhead=comp_tok + overhead_tokens + gen_tok_cap)
    
    # Save the compressed chapter for inspection and debugging
    compressed_path = output_path + ".compressed.txt"
    with open(compressed_path, 'w', encoding='utf-8') as comp_f:
        comp_f.write(chapter_text)
    print(f"Saved compressed chapter to: {compressed_path}")
    
    # Prepare prompts
    user_prompt = get_user_prompt(chapter_text, num_qa_pairs)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    prompt_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    
    # Generate response
    start_time = time.time()
    
    # SANITY CHECK: If the compressed text is too small, it likely means our compression was too aggressive
    if comp_tok < 100:
        print(f"WARNING: Compressed text is extremely small ({comp_tok} tokens).")
        print(f"Generating a minimal set of entries for the core facts only.")
        
        # Generate a small set of entries with a minimal prompt instead
        minimal_prompt = f"""Generate {num_qa_pairs} multiple-choice quiz entries about the Mahābhārata in XML format:
<entry>
  <reasoning>Brief justification for the answer</reasoning>
  <question>Question about a key character or event</question>
  <answer>Correct option (must be one of the options)</answer>
</entry>

Make sure questions are factually accurate based on the epic."""
        
        try:
            minimal_response = generate(model, tokenizer, prompt=minimal_prompt, max_tokens=min(4096, gen_tok_cap))
            
            with open(output_path, 'w', encoding='utf-8') as out_f:
                out_f.write(minimal_response)
            print(f"Generated minimal output for {os.path.basename(chunk_path)}")
            return True
        except Exception as e:
            print(f"ERROR generating minimal entries: {e}")
            return False
    
    # Normal generation path when compression worked properly
    generation_args = {
        "max_tokens": gen_tok_cap,
        "prompt": prompt_input,
    }
    
    # Add sampling parameters if they're supported
    try:
        response = generate(model, tokenizer, **generation_args)
    except Exception as e:
        # If there's still an error, try with default parameters only
        print(f"ERROR during generation: {type(e).__name__}: {e}")
        try:
            print("Attempting fallback generation with simplified parameters...")
            response = generate(model, tokenizer, prompt=prompt_input, max_tokens=min(2048, gen_tok_cap))
        except Exception as e2:
            print(f"CRITICAL: Second generation attempt failed: {type(e2).__name__}: {e2}")
            print("Writing error information to output file and exiting.")
            with open(output_path + ".error.txt", "w", encoding="utf-8") as err_f:
                err_f.write(f"First error: {type(e).__name__}: {e}\n\nSecond error: {type(e2).__name__}: {e2}")
            return False
    
    generation_time = time.time() - start_time
    
    # Save raw response for debugging
    with open(output_path + ".raw.txt", "w", encoding="utf-8") as raw_f:
        raw_f.write(response)

    # Strip optional <think> block the model may prepend
    response_clean = re.sub(r"<think>[\s\S]*?</think>", "", response, flags=re.IGNORECASE).strip()
    if not response_clean:
        response_clean = response  # fallback if regex failed
    # Save cleaned output
    with open(output_path, 'w', encoding='utf-8') as out_f:
        out_f.write(response_clean)
    
    print(f"Generated output for {os.path.basename(chunk_path)} in {generation_time:.2f}s | cleaned chars: {len(response_clean)}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Generate Q&A pairs from text chunks using Qwen3')
    parser.add_argument('--model_name', '-m', default="mlx-community/Qwen3-30B-A3B-4bit", 
                      help='MLX model repository name')
    parser.add_argument('--chunks_dir', '-c', default="../data/chunks", 
                      help='Directory containing text chunks')
    parser.add_argument('--output_dir', '-o', default="../data/outputs", 
                      help='Directory to save model outputs')
    parser.add_argument('--num_qa_pairs', '-n', type=int, default=2, 
                      help='Number of QA pairs to generate per chunk')
    parser.add_argument('--max_tokens', type=int, default=1024, 
                      help='Maximum tokens to generate per response')
    parser.add_argument('--temperature', '-t', type=float, default=0.7, 
                      help='Sampling temperature')
    parser.add_argument('--top_p', '-p', type=float, default=0.9, 
                      help='Nucleus sampling probability')
    parser.add_argument('--single_chunk', type=str, default=None,
                      help='Process only a single chunk file (for parallel processing)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}...")
    model, tokenizer = load(
        args.model_name,
        tokenizer_config={"trust_remote_code": True},
    )
    print("Model loaded. Starting generation...")
    
    # Dynamically detect context window of the tokenizer / model
    global MODEL_CTX
    try:
        # Get model context size but cap at our MAX_SAFE_CONTEXT
        detected_ctx = getattr(tokenizer, "model_max_length", 0)
        if detected_ctx > 0:
            MODEL_CTX = min(detected_ctx, MAX_SAFE_CONTEXT)
            print(f"[DEBUG] Tokenizer reported max length: {detected_ctx}, capped to {MODEL_CTX}")
        else:
            print(f"[DEBUG] Couldn't detect model context size, using default: {MODEL_CTX}")
    except Exception as e:
        print(f"[DEBUG] Error detecting model context size: {e}, using default: {MODEL_CTX}")

    # Process either a single chunk or all chunks
    if args.single_chunk:
        # Process only the specified chunk
        if not os.path.exists(os.path.join(args.chunks_dir, args.single_chunk)):
            print(f"Error: Chunk file {args.single_chunk} not found")
            return
        
        chunk_path = os.path.join(args.chunks_dir, args.single_chunk)
        chunk_id = args.single_chunk.split('.')[0]  # e.g., "chunk_0001"
        output_path = os.path.join(args.output_dir, f"{chunk_id}_out.xml")
        
        print(f"Processing chunk: {args.single_chunk}")
        generate_qa_pairs(
            model, tokenizer, chunk_path, output_path,
            num_qa_pairs=args.num_qa_pairs,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
    else:
        # Process all chunks
        chunk_files = sorted([f for f in os.listdir(args.chunks_dir) if f.endswith('.txt')])
        total_chunks = len(chunk_files)
        processed = 0
        
        for i, filename in enumerate(chunk_files):
            chunk_path = os.path.join(args.chunks_dir, filename)
            chunk_id = filename.split('.')[0]  # e.g., "chunk_0001"
            output_path = os.path.join(args.output_dir, f"{chunk_id}_out.xml")
            
            print(f"Processing {i+1}/{total_chunks}: {filename}")
            success = generate_qa_pairs(
                model, tokenizer, chunk_path, output_path,
                num_qa_pairs=args.num_qa_pairs,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p
            )
            
            if success:
                processed += 1
        
        print(f"Generation complete. Processed {processed}/{total_chunks} chunks.")

if __name__ == "__main__":
    main() 