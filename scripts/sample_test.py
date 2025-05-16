"""
Sample text generator for testing the Q&A generation pipeline.
"""

import os
import random
import argparse

# Some snippet of Sanskrit text from the Mahabharata
SAMPLE_TEXT = """
अथ द्वादशवर्षाणि पाण्डवा वनवासिनः।
अज्ञातवासं च परं वर्षमेकं प्रवेक्ष्यन्ति।।
वनवासो ह्ययं तेषां विविधैः क्लेशकारकैः।
अभवज्जनसंपूर्णे राज्ये च शुभकारिणि।।

धर्मराजो युधिष्ठिरः कुन्तीपुत्रो महायशाः।
अर्जुनश्च महाबाहुर्भीमसेनोऽतिकोपनः।।
नकुलः सहदेवश्च कुन्तीपुत्रा महाबलाः।
द्रौपदी च महापद्मा पाञ्चाली सत्यवादिनी।।

एते च वनवासिनः पाण्डवाः सह द्रौपद्या।
आसन्यथा महारण्ये महाकष्टं सहिष्णवः।।
अमृष्यमाणा वासं तं ब्राह्मणानां सहायिनः।
आश्रमे रमणीये ते न्यवसन्संयतेन्द्रियाः।।

तेषां तु पुत्राः पाञ्चाल्याः पाण्डवानां महात्मनाम्।
अभिमन्युश्च विख्यातो ब्रह्मचारी महाबलः।।
यौवनाश्वस्तथा मत्स्यः श्रुतकीर्तिर्महारथः।
शतानीकश्च विख्यातः श्रुतसेनो महाबलः।।

एते तु पाण्डुपुत्राणां पुत्राः प्रख्यातविक्रमाः।
विराटनगरे गुप्ताः कुर्वन्तो वीर्यकर्माणि।।
"""

def generate_sample_text(output_path, size=5000):
    """Generate a sample text file with repeated Sanskrit text."""
    # Calculate how many repetitions needed to reach the size
    repetitions = max(1, size // len(SAMPLE_TEXT))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        # Repeat the text to reach the desired size
        for _ in range(repetitions):
            f.write(SAMPLE_TEXT)
            # Add some random spacing/formatting for variety
            if random.random() > 0.7:
                f.write("\n\n")
            else:
                f.write("\n")
    
    print(f"Generated sample text of approximately {repetitions * len(SAMPLE_TEXT)} characters at {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate sample Sanskrit text for testing')
    parser.add_argument('--output', '-o', default='data/raw/sample_mahabharata.txt',
                        help='Path to save the generated text')
    parser.add_argument('--size', '-s', type=int, default=50000,
                        help='Approximate size of the sample text in characters')
    
    args = parser.parse_args()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Generate the sample text
    generate_sample_text(args.output, args.size)
    
    print("Sample text generated. You can now run the pipeline:")
    print(f"1. python chunk_data.py --input {args.output}")
    print(f"2. python generate_xml.py or parallel_generate.py")
    print(f"3. python parse_and_format.py")

if __name__ == "__main__":
    main() 