import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pdfplumber
from pydantic import BaseModel
import numpy as np
from collections import defaultdict

class HeadingItem(BaseModel):
    level: str
    text: str
    page: int  # 0-indexed

class PDFOutline(BaseModel):
    title: str
    outline: List[HeadingItem]

def clean_text(text: str) -> str:
    """Advanced text normalization that preserves structure"""
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)  # Remove control chars
    text = re.sub(r'(?<!\n)\s+', ' ', text)  # Normalize spaces
    return text.strip()

class PDFStructureExtractor:
    def __init__(self):
        self.font_stats = defaultdict(list)
        self.median_size = 12  # Default fallback

    def analyze_fonts(self, page):
        """Collect font statistics without bold detection"""
        words = page.extract_words(extra_attrs=["size", "fontname"])
        for word in words:
            if word["text"].strip():
                self.font_stats[word["fontname"]].append(word["size"])
        return words

    def compute_typography(self):
        """Calculate dominant font characteristics"""
        all_sizes = []
        for sizes in self.font_stats.values():
            all_sizes.extend(sizes)
        if all_sizes:
            self.median_size = np.median(all_sizes)

    def detect_heading_level(self, text: str, font_size: float) -> Optional[str]:
        """Reliable heading detection using multiple heuristics"""
        clean_txt = clean_text(text)
        
        # Skip obvious non-headings
        if len(clean_txt) > 100 or len(clean_txt.split()) > 8:
            return None
        
        # Size-based detection
        size_ratio = font_size / self.median_size if self.median_size else 1
        
        # Content patterns (ordered by specificity)
        patterns = [
            (r'^(?:APPENDIX|Appendix|CHAPTER|Chapter|SECTION|Section)\b', "H1"),
            (r'^\d+\.\d+\.\d+\s', "H3"),
            (r'^\d+\.\d+\s', "H2"),
            (r'^\d+\.\s', "H2"),
            (r'^[IVXLCDM]+\.', "H1"),
            (r'^[A-Z][A-Z0-9\s]+:$', "H2"),
            (r'^[A-Z][a-z]+:', "H3"),
            (r'^•\s', "H4")
        ]
        
        for pattern, level in patterns:
            if re.match(pattern, clean_txt):
                return level
        
        # Size heuristics (more conservative)
        if size_ratio > 2.0:
            return "H1" if len(clean_txt.split()) <= 5 else "H2"
        elif size_ratio > 1.5:
            return "H2" if clean_txt.endswith(':') else "H3"
        
        return None

def extract_pdf_structure(pdf_path: Path) -> Dict:
    """Robust PDF processing with comprehensive error handling"""
    extractor = PDFStructureExtractor()
    result = {"title": "", "outline": []}
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # First pass: document analysis
            for page in pdf.pages:
                extractor.analyze_fonts(page)
            extractor.compute_typography()
            
            # Second pass: content extraction
            for page_num, page in enumerate(pdf.pages):
                words = page.extract_words(extra_attrs=["size", "fontname"])
                current_line = ""
                current_size = None
                
                for word in words:
                    word_text = clean_text(word["text"])
                    if not word_text:
                        continue
                    
                    # New line on significant size change
                    if current_size and abs(word["size"] - current_size) > 2.0:
                        if current_line:
                            level = extractor.detect_heading_level(current_line, current_size)
                            if level:
                                result["outline"].append({
                                    "level": level,
                                    "text": current_line,
                                    "page": page_num
                                })
                        current_line = word_text
                        current_size = word["size"]
                    else:
                        current_line += (" " + word_text) if current_line else word_text
                        current_size = current_size or word["size"]
                
                # Process remaining content
                if current_line:
                    level = extractor.detect_heading_level(current_line, current_size)
                    if level:
                        result["outline"].append({
                            "level": level,
                            "text": current_line,
                            "page": page_num
                        })
            
            # Determine title
            if not result["title"]:
                first_page = pdf.pages[0].extract_text()
                if first_page:
                    result["title"] = clean_text(first_page.split('\n')[0])
    
    except Exception as e:
        print(f"Error processing {pdf_path.name}: {str(e)}")
        return {"title": f"Error: {str(e)}", "outline": []}
    
    # Post-processing
    seen = set()
    final_outline = []
    for item in result["outline"]:
        # Use first 40 chars + page as fingerprint
        fingerprint = f"{item['page']}-{item['text'][:40]}"
        if fingerprint not in seen:
            final_outline.append(item)
            seen.add(fingerprint)
    
    result["outline"] = final_outline
    return result

def process_pdfs():
    base_input_dir = Path("./input")

    for collection_dir in base_input_dir.glob("Collection_*"):
        pdf_dir = collection_dir / "PDFs"
        output_dir = collection_dir / "Processed_pdf"
        output_dir.mkdir(parents=True, exist_ok=True)

        if not pdf_dir.exists():
            print(f"Skipping {collection_dir.name} — No PDFs directory found.")
            continue

        for pdf_file in pdf_dir.glob("*.pdf"):
            print(f"Processing {pdf_file.name} in {collection_dir.name}...")
            structure = extract_pdf_structure(pdf_file)

            output_file = output_dir / f"{pdf_file.stem}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(PDFOutline(**structure).dict(), f, indent=2)


if __name__ == "__main__":
    print("=== Starting PDF Processor ===")
    print(f"Current directory: {Path.cwd()}")
    print(f"Collections found: {[d.name for d in Path('./input').glob('Collection_*')]}")
    process_pdfs()