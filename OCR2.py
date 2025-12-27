import pytesseract
from PIL import Image
import requests
import re
import sys

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from tkinter import Tk, filedialog

from tkinter import Tk, filedialog

# Hide the root window
root = Tk()
root.withdraw()

# Open file dialog
IMAGE_PATH = filedialog.askopenfilename(
    title="Select an Image",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.gif")]
)

# ✅ UPDATED OLLAMA DETAILS
OLLAMA_URL = "https://curly-orbit-q5v54pwrg4g24wj5-11434.app.github.dev/api/generate"
MODEL_NAME = "nuextract:3.8b"


# --------------------------------------------------
# 1. OCR (Optical Character Recognition)
# --------------------------------------------------
def extract_text_from_image(image_path):
    try:
        img = Image.open(image_path)
        # --psm 3 is default (auto page segmentation), often best for mixed IDs
        # preserve_interword_spaces helps keep 'Name' and 'Value' visually separated
        config = r'--oem 3 --psm 3 -c preserve_interword_spaces=1'
        text = pytesseract.image_to_string(img, lang="eng", config=config)
        return text
    except FileNotFoundError:
        print(f"❌ Error: File '{image_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ OCR Error: {e}")
        sys.exit(1)


# --------------------------------------------------
# 2. INTELLIGENT CLEANING (Fixes the "Cutting" Issue)
# --------------------------------------------------
def clean_ocr_text(raw_text):
    lines = raw_text.splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        
        # If line is empty, skip
        if not line:
            continue

        # ✅ CRITICAL FIX: Allow more punctuation characters
        # We keep: Letters, Numbers, Spaces, Dots (.), Commas (,), Dashes (-), 
        # Slashes (/), Colons (:), and Brackets ()
        # This prevents breaking names like "Dr. A.K. Singh" or formatting.
        line = re.sub(r"[^A-Za-z0-9\s\.\,\-\/\:\(\)]", "", line)

        # Remove extra spaces inside the line (e.g., "Name    :" becomes "Name :")
        line = re.sub(r'\s+', ' ', line).strip()

        # ✅ CRITICAL FIX: Lower length threshold
        # Only skip if it's 1 char (noise). 
        # We keep 2+ chars to preserve initials or "Male"/"Female" markers.
        if len(line) < 2:
            continue

        # ✅ DELETED THE "GARBAGE WORDS" LIST
        # We now send "Government of India" etc. to the AI. 
        # The AI needs these headers to know which document it is looking at.
        
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


# --------------------------------------------------
# 3. SEND TO OLLAMA (NuExtract Optimized)
# --------------------------------------------------
def send_to_ollama(clean_text):
    # We give the model a stricter schema to ensure it picks the right data
    prompt = f"""
### Template:
Document Type: <"PAN Card" or "Aadhaar Card">
PAN Number: <extract if present>
Aadhaar Number: <extract if present, format XXXX XXXX XXXX>
Name: <extract name of the person>
Father's Name: <extract father's name if present>
Date of Birth: <DD/MM/YYYY format>
Gender: <Male/Female>
ID Number: <If PAN or Aadhaar specific number found>

### Input Text:
{clean_text[:3000]}

### Instructions:
Analyze the Input Text above.
1. Identify if it is a PAN Card or Aadhaar Card.
2. Extract the Name, ID Number, DOB, and Gender accurately.
3. Ignore government headers (like "Govt of India" or "Income Tax Dept") BUT use them to confirm document type.
4. Do not make up information. If a field is missing, leave it blank.
"""

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0, # Deterministic (Fact-based)
            "num_ctx": 2048
        }
    }

    try:
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=300
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        return f"Error connecting to Ollama: {e}"


# --------------------------------------------------
# 4. MAIN EXECUTION
# --------------------------------------------------
def main():
    print("===================================")
    print(f"🚀 Model: {MODEL_NAME}")
    print("===================================")

    print(f"\n🔍 Reading '{IMAGE_PATH}'...")
    raw_text = extract_text_from_image(IMAGE_PATH)

    print("\n📄 RAW OCR OUTPUT (First 300 chars):")
    print("-" * 30)
    print(raw_text[:300] + "...")
    print("-" * 30)

    print("\n🧹 Smart Cleaning Text...")
    clean_text = clean_ocr_text(raw_text)

    # Debug: Show user exactly what is going to the AI
    print("\n📘 CLEAN TEXT SENT TO AI:")
    print("-" * 30)
    print(clean_text)
    print("-" * 30)

    print("\n⚡ Extracting details via Ollama...")
    result = send_to_ollama(clean_text)

    print("\n✅ FINAL RESULT:\n")
    print(result)

    # Save
    with open("extracted_details.txt", "w", encoding="utf-8") as f:
        f.write(result)
    print("\n💾 Saved to extracted_details.txt")


if __name__ == "__main__":
    main()