import os
import json
import re
import argparse
import xml.etree.ElementTree as ET

def clean_xml(xml_text):
    """
    Clean the XML text for proper parsing:
    1. Remove <think> tags and their content
    2. Ensure there's a root element
    """
    # Remove <think>...</think> segments
    cleaned_xml = re.sub(r'<think>.*?</think>', '', xml_text, flags=re.DOTALL)
    
    # Check if there's a root element (entry or entries)
    if cleaned_xml.strip().startswith("<entry"):
        # Add a root element if we have multiple entries
        if cleaned_xml.count("<entry>") > 1 or cleaned_xml.count("<entry ") > 1:
            xml_str = f"<entries>{cleaned_xml}</entries>"
        else:
            xml_str = cleaned_xml
    else:
        # If model didn't include <entry> tag, add it (fallback)
        xml_str = f"<entry>{cleaned_xml}</entry>"
    
    return xml_str

def parse_xml_output(xml_path):
    """Parse an XML file and extract question-answer pairs"""
    try:
        with open(xml_path, 'r', encoding='utf-8') as f:
            xml_text = f.read().strip()
        
        if not xml_text:
            print(f"Empty file: {xml_path}")
            return []
        
        # Clean the XML
        try:
            xml_str = clean_xml(xml_text)
            root = ET.fromstring(xml_str)
        except ET.ParseError as e:
            print(f"XML parse error in {os.path.basename(xml_path)}: {e}")
            # Try a more aggressive cleaning approach for malformed XML
            try:
                # Extract just question and answer tags with regex as fallback
                questions = re.findall(r'<question>(.*?)</question>', xml_text, re.DOTALL)
                answers = re.findall(r'<answer>(.*?)</answer>', xml_text, re.DOTALL)
                
                qa_pairs = []
                for i in range(min(len(questions), len(answers))):
                    qa_pairs.append({
                        "question": questions[i].strip(),
                        "answer": answers[i].strip()
                    })
                return qa_pairs
            except Exception as e2:
                print(f"Failed to extract with regex too: {e2}")
                return []
        
        # Extract QA pairs from parsed XML
        qa_pairs = []
        
        # Handle both single entry and multiple entries
        entries = root.findall('.//entry') if root.tag != 'entry' else [root]
        
        for entry in entries:
            # Find all question-answer pairs in this entry
            questions = entry.findall('.//question')
            answers = entry.findall('.//answer')
            
            # Match questions with answers
            for i in range(min(len(questions), len(answers))):
                question_text = questions[i].text
                answer_text = answers[i].text
                
                if question_text and answer_text:  # Skip if either is None
                    qa_pairs.append({
                        "question": question_text.strip(),
                        "answer": answer_text.strip()
                    })
        
        return qa_pairs
    
    except Exception as e:
        print(f"Error processing {os.path.basename(xml_path)}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description='Parse XML outputs and create JSONL dataset')
    parser.add_argument('--input_dir', '-i', default="../data/outputs", 
                      help='Directory containing XML outputs')
    parser.add_argument('--output_file', '-o', default="../data/final/mahabharata_qa.jsonl", 
                      help='Output JSONL file path')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Process all XML files
    xml_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.xml')])
    total_files = len(xml_files)
    total_qa_pairs = 0
    
    with open(args.output_file, 'w', encoding='utf-8') as jsonl_f:
        for i, filename in enumerate(xml_files):
            xml_path = os.path.join(args.input_dir, filename)
            print(f"Processing {i+1}/{total_files}: {filename}")
            
            qa_pairs = parse_xml_output(xml_path)
            for qa_pair in qa_pairs:
                jsonl_f.write(json.dumps(qa_pair, ensure_ascii=False) + "\n")
                total_qa_pairs += 1
    
    print(f"Parsing complete. Extracted {total_qa_pairs} QA pairs from {total_files} files.")
    print(f"Output saved to: {args.output_file}")

if __name__ == "__main__":
    main() 