import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import requests
import tkinter as tk
from tkinter import filedialog
import re
import cv2
import numpy as np

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

OLLAMA_URL = "https://curly-orbit-q5v54pwrg4g24wj5-11434.app.github.dev/api/generate"
MODEL_NAME = "phi3:mini"


# --------------------------------------------------
# IMAGE PREPROCESSING
# --------------------------------------------------
def preprocess_image(image_path):
    """Enhanced image preprocessing for better OCR"""
    # Read image with OpenCV
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    
    # Convert back to PIL Image
    pil_image = Image.fromarray(denoised)
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(pil_image)
    enhanced = enhancer.enhance(2.0)
    
    # Sharpen
    sharpened = enhanced.filter(ImageFilter.SHARPEN)
    
    return sharpened


# --------------------------------------------------
# IMAGE PICKER
# --------------------------------------------------
def pick_image():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select PAN / Aadhaar Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )
    return file_path


# --------------------------------------------------
# ENHANCED OCR WITH MULTIPLE CONFIGS
# --------------------------------------------------
def extract_text(image_path):
    """Extract text using multiple OCR configurations"""
    print("📸 Preprocessing image...")
    processed_img = preprocess_image(image_path)
    
    # Configuration 1: Standard
    config1 = '--oem 3 --psm 6'
    text1 = pytesseract.image_to_string(processed_img, lang="eng", config=config1)
    
    # Configuration 2: Treat as single block
    config2 = '--oem 3 --psm 4'
    text2 = pytesseract.image_to_string(processed_img, lang="eng", config=config2)
    
    # Configuration 3: Original image without preprocessing
    original_img = Image.open(image_path)
    text3 = pytesseract.image_to_string(original_img, lang="eng", config=config1)
    
    # Combine all extractions
    combined_text = f"{text1}\n{text2}\n{text3}"
    
    return combined_text


# --------------------------------------------------
# PATTERN MATCHING FOR AADHAAR & PAN
# --------------------------------------------------
def extract_with_regex(text):
    """Extract Aadhaar and PAN using regex patterns"""
    results = {
        "aadhaar": None,
        "pan": None,
        "confidence": "low"
    }
    
    # Clean text - remove extra spaces and normalize
    cleaned_text = re.sub(r'\s+', ' ', text)
    
    # Aadhaar patterns (12 digits, may have spaces)
    # Patterns: XXXX XXXX XXXX or XXXXXXXXXXXX or XXXX-XXXX-XXXX
    aadhaar_patterns = [
        r'\b\d{4}\s\d{4}\s\d{4}\b',  # Format: XXXX XXXX XXXX
        r'\b\d{12}\b',                # Format: XXXXXXXXXXXX
        r'\b\d{4}-\d{4}-\d{4}\b',    # Format: XXXX-XXXX-XXXX
    ]
    
    for pattern in aadhaar_patterns:
        match = re.search(pattern, cleaned_text)
        if match:
            aadhaar_num = match.group().replace(' ', '').replace('-', '')
            # Validate it's exactly 12 digits
            if len(aadhaar_num) == 12 and aadhaar_num.isdigit():
                results["aadhaar"] = aadhaar_num
                results["confidence"] = "high"
                break
    
    # PAN patterns (5 letters + 4 digits + 1 letter)
    # Format: ABCDE1234F
    pan_patterns = [
        r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b',  # Standard format
        r'\b[A-Z]{3}P[A-Z][0-9]{4}[A-Z]\b',  # Personal PAN (4th char is P)
    ]
    
    # Also search without word boundaries for cases with noise
    text_upper = text.upper()
    for pattern in pan_patterns:
        match = re.search(pattern, text_upper)
        if match:
            results["pan"] = match.group()
            results["confidence"] = "high"
            break
    
    # If no match with strict patterns, try relaxed search
    if not results["pan"]:
        # Look for sequences that might be PAN with OCR errors
        relaxed_pan = re.search(r'[A-Z0-9]{10}', text_upper)
        if relaxed_pan:
            candidate = relaxed_pan.group()
            # Check if it roughly matches PAN pattern
            if (candidate[0:5].isalpha() and 
                candidate[5:9].isdigit() and 
                candidate[9].isalpha()):
                results["pan"] = candidate
                results["confidence"] = "medium"
    
    return results


# --------------------------------------------------
# SEND TO OLLAMA (WITH BETTER PROMPT)
# --------------------------------------------------
def extract_with_ollama(ocr_text):
    """Use LLM as backup extraction method"""
    prompt = f"""
You are an expert at extracting Indian ID numbers from OCR text.

TASK: Extract ONLY the Aadhaar number OR PAN number from the text below.

RULES:
1. Aadhaar: Exactly 12 digits (may have spaces or dashes in between)
2. PAN: Exactly 10 characters (5 letters + 4 digits + 1 letter)
3. Return ONLY the number, nothing else
4. If both found, return both on separate lines starting with "AADHAAR:" or "PAN:"
5. If nothing found, return "NOT_FOUND"
6. Remove all spaces and dashes from Aadhaar

TEXT:
{ocr_text[:3000]}

RESPONSE (number only):
"""

    try:
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9
            }
        }

        response = requests.post(OLLAMA_URL, json=payload, timeout=90)
        response.raise_for_status()

        result = response.json()["response"].strip()
        return result
    except Exception as e:
        print(f"⚠️ Ollama extraction failed: {e}")
        return "NOT_FOUND"


# --------------------------------------------------
# MAIN EXTRACTION LOGIC
# --------------------------------------------------
def extract_id_numbers(ocr_text):
    """Combined extraction using regex first, then Ollama"""
    print("\n🔍 Method 1: Pattern Matching (Regex)...")
    regex_results = extract_with_regex(ocr_text)
    
    # If regex found something with high confidence, return it
    if regex_results["confidence"] == "high":
        print("✅ Found with high confidence using pattern matching!")
        return regex_results
    
    # Otherwise, try Ollama
    print("\n🤖 Method 2: AI Extraction (Ollama)...")
    ollama_result = extract_with_ollama(ocr_text)
    
    # Parse Ollama result
    if "NOT_FOUND" not in ollama_result.upper():
        # Check if Ollama found Aadhaar
        aadhaar_match = re.search(r'AADHAAR:\s*(\d{12})', ollama_result)
        if aadhaar_match:
            regex_results["aadhaar"] = aadhaar_match.group(1)
            regex_results["confidence"] = "medium"
        
        # Check if Ollama found PAN
        pan_match = re.search(r'PAN:\s*([A-Z]{5}\d{4}[A-Z])', ollama_result)
        if pan_match:
            regex_results["pan"] = pan_match.group(1)
            regex_results["confidence"] = "medium"
        
        # If result doesn't have labels, try to identify
        if not aadhaar_match and not pan_match:
            clean_result = ollama_result.strip().replace(' ', '').replace('-', '')
            if len(clean_result) == 12 and clean_result.isdigit():
                regex_results["aadhaar"] = clean_result
                regex_results["confidence"] = "medium"
            elif len(clean_result) == 10 and clean_result[0:5].isalpha():
                regex_results["pan"] = clean_result.upper()
                regex_results["confidence"] = "medium"
    
    return regex_results


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------
def display_results(results):
    """Pretty print the extracted results"""
    print("\n" + "="*60)
    print("📋 EXTRACTION RESULTS")
    print("="*60)
    
    if results["aadhaar"]:
        formatted_aadhaar = f"{results['aadhaar'][:4]} {results['aadhaar'][4:8]} {results['aadhaar'][8:]}"
        print(f"\n✅ AADHAAR NUMBER FOUND:")
        print(f"   {formatted_aadhaar}")
    
    if results["pan"]:
        print(f"\n✅ PAN NUMBER FOUND:")
        print(f"   {results['pan']}")
    
    if not results["aadhaar"] and not results["pan"]:
        print("\n❌ NO VALID ID NUMBER FOUND")
        print("   Please ensure:")
        print("   - Image is clear and well-lit")
        print("   - Numbers are not obscured")
        print("   - Image is not blurry or skewed")
    
    print(f"\n📊 Confidence: {results['confidence'].upper()}")
    print("="*60)


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    print("="*60)
    print("🇮🇳 AADHAAR / PAN CARD NUMBER EXTRACTOR")
    print("="*60)
    
    image_path = pick_image()

    if not image_path:
        print("\n❌ No image selected. Exiting...")
        return

    print(f"\n🖼️  Image selected: {image_path}")

    print("\n🔄 Running Enhanced OCR...")
    ocr_text = extract_text(image_path)
    
    print("\n📄 EXTRACTED TEXT:")
    print("-" * 60)
    print(ocr_text[:500] + ("..." if len(ocr_text) > 500 else ""))
    print("-" * 60)

    print("\n🚀 Extracting ID numbers...")
    results = extract_id_numbers(ocr_text)
    
    display_results(results)
    
    # Save results to file
    if results["aadhaar"] or results["pan"]:
        with open("extracted_ids.txt", "w") as f:
            f.write("EXTRACTION RESULTS\n")
            f.write("="*40 + "\n\n")
            if results["aadhaar"]:
                f.write(f"Aadhaar: {results['aadhaar']}\n")
            if results["pan"]:
                f.write(f"PAN: {results['pan']}\n")
            f.write(f"\nConfidence: {results['confidence']}\n")
        print("\n💾 Results saved to 'extracted_ids.txt'")


# --------------------------------------------------
# RUN
# --------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()