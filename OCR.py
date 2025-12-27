import pytesseract
from PIL import Image
import requests
import re
import sys
import time
import os

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

# Auto-detect Tesseract installation path
def find_tesseract():
    """Automatically find Tesseract installation"""
    possible_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
        "/opt/homebrew/bin/tesseract"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # Try system PATH
    try:
        import shutil
        tesseract_path = shutil.which("tesseract")
        if tesseract_path:
            return tesseract_path
    except:
        pass
    
    return None

# Set Tesseract path
tesseract_path = find_tesseract()
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
    print(f"✅ Found Tesseract at: {tesseract_path}")
else:
    print("⚠️ Tesseract not found. Please install it:")
    print("   Windows: https://github.com/UB-Mannheim/tesseract/wiki")
    print("   Linux: sudo apt install tesseract-ocr")
    print("   Mac: brew install tesseract")

IMAGE_PATH = "imgg.jpg"
OLLAMA_URL = "https://curly-orbit-q5v54pwrg4g24wj5-11434.app.github.dev/api/generate"
MODEL_NAME = "llama3.2:1b"

# --------------------------------------------------
# 1. LIGHTNING FAST OCR
# --------------------------------------------------
def run_fast_ocr(image_path):
    """Extract text from image using optimized OCR"""
    try:
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"❌ Error: Image file '{image_path}' not found!")
            print(f"   Current directory: {os.getcwd()}")
            print(f"   Please place your image file in this directory.")
            sys.exit(1)
        
        # Open and optimize image
        img = Image.open(image_path)
        
        # Optimization 1: Resize if too big
        if img.width > 1200:
            ratio = 1200 / float(img.width)
            new_h = int(img.height * ratio)
            img = img.resize((1200, new_h), Image.Resampling.LANCZOS)
        
        # Optimization 2: Convert to Greyscale
        img = img.convert('L')
        
        # Optimization 3: PSM 6 for faster processing
        text = pytesseract.image_to_string(
            img, 
            lang='eng', 
            config='--oem 3 --psm 6'
        )
        
        return text
    
    except pytesseract.TesseractNotFoundError:
        print("❌ Tesseract is not installed or not found in PATH")
        print("   Please install Tesseract OCR:")
        print("   Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        print("   Linux: sudo apt install tesseract-ocr")
        print("   Mac: brew install tesseract")
        sys.exit(1)
    
    except Exception as e:
        print(f"❌ OCR Failed: {e}")
        sys.exit(1)

# --------------------------------------------------
# 2. PRE-CLEANING
# --------------------------------------------------
def clean_text(raw_text):
    """Clean and normalize OCR output"""
    if not raw_text or not raw_text.strip():
        return "No text detected"
    
    # Remove noise, keep essential characters
    lines = [
        re.sub(r"[^A-Za-z0-9\s\.\,\-\/\:]", "", line).strip() 
        for line in raw_text.splitlines()
    ]
    
    # Remove empty or very short lines
    valid_lines = [l for l in lines if len(l) > 2]
    
    return "\n".join(valid_lines) if valid_lines else "No valid text after cleaning"

# --------------------------------------------------
# 3. AI SORTING
# --------------------------------------------------
def organize_with_ollama(cleaned_text):
    """Use Ollama AI to extract and format information"""
    
    # Check if text is valid
    if not cleaned_text or cleaned_text.startswith("No"):
        return f"⚠️ Cannot process: {cleaned_text}"
    
    # Minimal prompt for fast processing
    prompt = f"""Input Data:
{cleaned_text[:1000]}

Task: Identify document type (PAN or Aadhaar) and extract details.

Format your response as:
Type: [PAN/Aadhaar]
ID No: [number]
Name: [full name]
Father: [father's name]
DOB: [date of birth]
"""
    
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_ctx": 1024,
            "num_predict": 100
        }
    }
    
    try:
        print(f"   Connecting to Ollama at: {OLLAMA_URL}")
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json().get("response", "").strip()
        
        if not result:
            return "⚠️ AI returned empty response. Check if Ollama is running and model is installed."
        
        return result
    
    except requests.exceptions.Timeout:
        return f"❌ Timeout: The model took too long (>60s).\n   Try: ollama pull {MODEL_NAME}"
    
    except requests.exceptions.ConnectionError:
        return f"❌ Connection Error: Cannot reach Ollama server.\n   URL: {OLLAMA_URL}\n   Make sure Ollama is running!"
    
    except requests.exceptions.HTTPError as e:
        return f"❌ HTTP Error: {e}\n   Check if model '{MODEL_NAME}' is installed: ollama pull {MODEL_NAME}"
    
    except Exception as e:
        return f"❌ Unexpected Error: {e}"

# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------
def main():
    """Main execution flow"""
    print("\n" + "="*50)
    print("🚀 FAST OCR + AI DOCUMENT PROCESSOR")
    print("="*50 + "\n")
    
    start_time = time.time()
    
    # Step 1: OCR
    print("1️⃣  Scanning Image...", end=" ", flush=True)
    raw_ocr = run_fast_ocr(IMAGE_PATH)
    cleaned_ocr = clean_text(raw_ocr)
    print("✓ Done")
    
    # Show preview of OCR output
    print(f"   Preview: {cleaned_ocr[:100]}...")
    
    # Step 2: AI Processing
    print("\n2️⃣  AI Sorting...", end=" ", flush=True)
    final_output = organize_with_ollama(cleaned_ocr)
    print("✓ Done")
    
    # Calculate time
    total_time = round(time.time() - start_time, 2)
    
    # Display results
    print(f"\n⏱️  Total Time: {total_time}s")
    print("\n" + "="*50)
    print("📄 EXTRACTED DETAILS:")
    print("="*50)
    print(final_output)
    print("="*50 + "\n")
    
    # Save to file
    try:
        with open("extracted_details.txt", "w", encoding="utf-8") as f:
            f.write(f"Processing Time: {total_time}s\n")
            f.write("="*50 + "\n")
            f.write(final_output)
        print("✅ Results saved to: extracted_details.txt\n")
    except Exception as e:
        print(f"⚠️  Could not save file: {e}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
        sys.exit(1)