# import json
# import re
# from pathlib import Path
# from typing import List, Dict, Optional, Tuple, Any
# import pdfplumber
# from pydantic import BaseModel
# import numpy as np
# from collections import defaultdict
# from datetime import datetime
# import spacy

# # Load a small spaCy model for NLP processing
# try:
#     nlp = spacy.load("en_core_web_sm")
# except OSError:
#     raise Exception("Please download the spaCy model: python -m spacy download en_core_web_sm")

# class DocumentSection(BaseModel):
#     document: str
#     page_number: int
#     section_title: str
#     content: str
#     importance_score: float = 0.0

# class OutputModel(BaseModel):
#     metadata: Dict[str, Any]
#     extracted_sections: List[Dict[str, Any]]
#     subsection_analysis: List[Dict[str, Any]]

# class PersonaAnalyzer:
#     def __init__(self, persona: str, job: str):
#         self.persona = persona.lower()
#         self.job = job.lower()
#         self.keywords = self._extract_keywords()
        
#     def _extract_keywords(self) -> List[str]:
#         """Extract relevant keywords from persona and job description"""
#         doc = nlp(f"{self.persona} {self.job}")
#         keywords = []
        
#         # Extract nouns and verbs
#         for token in doc:
#             if token.pos_ in ["NOUN", "PROPN", "VERB"] and not token.is_stop:
#                 keywords.append(token.lemma_.lower())
        
#         # Add domain-specific terms
#         if "researcher" in self.persona:
#             keywords.extend(["method", "result", "study", "data", "analysis"])
#         elif "student" in self.persona:
#             keywords.extend(["concept", "example", "definition", "theory"])
#         elif "analyst" in self.persona:
#             keywords.extend(["trend", "growth", "market", "financial", "revenue"])
            
#         return list(set(keywords))
    
#     def calculate_relevance(self, text: str) -> float:
#         """Calculate relevance score based on persona and job"""
#         if not text:
#             return 0.0
            
#         doc = nlp(text.lower())
#         matches = 0
#         total_keywords = len(self.keywords)
        
#         if total_keywords == 0:
#             return 0.0
            
#         for token in doc:
#             if token.lemma_ in self.keywords:
#                 matches += 1
                
#         # Normalize score
#         return min(matches / total_keywords * 10, 10.0)

# class PDFProcessor:
#     def __init__(self, persona: str, job: str):
#         self.persona_analyzer = PersonaAnalyzer(persona, job)
#         self.font_stats = defaultdict(list)
#         self.median_size = 12.0

#     def process_documents(self, pdf_paths: List[Path]) -> Dict:
#         """Process multiple PDFs and return structured results"""
#         results = []
        
#         for pdf_path in pdf_paths:
#             with pdfplumber.open(pdf_path) as pdf:
#                 for page_num, page in enumerate(pdf.pages):
#                     text = page.extract_text()
#                     if not text:
#                         continue
                        
#                     # Split into sections (simplified for example)
#                     sections = self._split_into_sections(text)
                    
#                     for section in sections:
#                         relevance = self.persona_analyzer.calculate_relevance(section)
#                         if relevance > 2.0:  # Threshold
#                             results.append({
#                                 "document": pdf_path.name,
#                                 "page_number": page_num + 1,
#                                 "section_title": self._extract_section_title(section),
#                                 "content": section,
#                                 "importance_score": relevance
#                             })
        
#         # Sort by importance
#         results.sort(key=lambda x: x["importance_score"], reverse=True)
        
#         return results
    
#     def _split_into_sections(self, text: str) -> List[str]:
#         """Simple section splitting heuristic"""
#         # Split by common section delimiters
#         sections = re.split(r'\n\s*\d+\.|\n\s*[A-Z][A-Z0-9\s]+:|\n\s*[A-Z][a-z]+:', text)
#         return [s.strip() for s in sections if s.strip()]
    
#     def _extract_section_title(self, text: str) -> str:
#         """Extract a title from section text"""
#         first_line = text.split('\n')[0]
#         return first_line[:100]  # Truncate if too long

# def generate_output(input_data: Dict, processed_sections: List[Dict]) -> Dict:
#     """Generate the required output format"""
#     metadata = {
#         "input_documents": [doc["filename"] for doc in input_data["documents"]],
#         "persona": input_data["persona"]["role"],
#         "job_to_be_done": input_data["job_to_be_done"]["task"],
#         "processing_timestamp": datetime.utcnow().isoformat()
#     }
    
#     extracted_sections = []
#     subsection_analysis = []
    
#     for section in processed_sections:
#         extracted_sections.append({
#             "document": section["document"],
#             "page_number": section["page_number"],
#             "section_title": section["section_title"],
#             "importance_rank": section["importance_score"]
#         })
        
#         subsection_analysis.append({
#             "document": section["document"],
#             "page_number": section["page_number"],
#             "refined_text": section["content"][:500],  # Truncate for example
#             "constraints": "None"  # Could add constraints based on content
#         })
    
#     return OutputModel(
#         metadata=metadata,
#         extracted_sections=extracted_sections,
#         subsection_analysis=subsection_analysis
#     ).dict()

# def main():
#     # Load input data
#     input_file = Path("./input/challenge1b_input.json")
#     with open(input_file) as f:
#         input_data = json.load(f)
    
#     # Process documents
#     pdf_paths = [Path("./input") / doc["filename"] for doc in input_data["documents"]]
#     processor = PDFProcessor(
#         persona=input_data["persona"]["role"],
#         job=input_data["job_to_be_done"]["task"]
#     )
#     processed_sections = processor.process_documents(pdf_paths)
    
#     # Generate output
#     output_data = generate_output(input_data, processed_sections)
    
#     # Save output
#     output_file = Path("./output/challenge1b_output.json")
#     output_file.parent.mkdir(exist_ok=True)
#     with open(output_file, "w") as f:
#         json.dump(output_data, f, indent=2)

# if __name__ == "__main__":
#     main()


# import json
# import re
# from pathlib import Path
# from typing import List, Dict, Optional, Tuple, Any
# import pdfplumber
# from pydantic import BaseModel
# import numpy as np
# from collections import defaultdict
# from datetime import datetime
# import spacy

# # Load a small English model for NLP processing
# nlp = spacy.load("en_core_web_sm")

# class Config(BaseModel):
#     challenge_info: Dict[str, str]  # Instead of separate fields
#     documents: List[Dict[str, str]]
#     persona: Dict[str, str]
#     job_to_be_done: Dict[str, str]

# class Section(BaseModel):
#     document: str
#     page_number: int
#     section_title: str
#     importance_rank: int

# class SubSection(BaseModel):
#     document: str
#     page_number: int
#     refined_text: str
#     constraints: List[str]

# class Output(BaseModel):
#     metadata: Dict[str, Any]
#     extracted_sections: List[Section]
#     subsection_analysis: List[SubSection]

# class HeadingItem(BaseModel):
#     level: str
#     text: str
#     page: int  # 0-indexed

# class PDFOutline(BaseModel):
#     title: str
#     outline: List[HeadingItem]

# def clean_text(text: str) -> str:
#     """Advanced text normalization that preserves structure"""
#     text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)  # Remove control chars
#     text = re.sub(r'(?<!\n)\s+', ' ', text)  # Normalize spaces
#     return text.strip()

# class PersonaAnalyzer:
#     def __init__(self, persona: str, job: str):
#         self.persona = persona.lower()
#         self.job = job.lower()
#         self.keywords = self._extract_keywords()
        
#     def _extract_keywords(self) -> List[str]:
#         """Extract relevant keywords from persona and job description"""
#         doc = nlp(f"{self.persona} {self.job}")
#         keywords = []
        
#         # Extract nouns and verbs
#         for token in doc:
#             if token.pos_ in ["NOUN", "PROPN", "VERB"] and not token.is_stop:
#                 keywords.append(token.lemma_.lower())
        
#         # Add domain-specific terms
#         if "researcher" in self.persona:
#             keywords.extend(["methodology", "dataset", "result", "analysis"])
#         elif "analyst" in self.persona:
#             keywords.extend(["trend", "investment", "strategy", "market"])
#         elif "student" in self.persona:
#             keywords.extend(["concept", "exam", "study", "key"])
            
#         return list(set(keywords))
    
#     def score_relevance(self, text: str) -> float:
#         """Score text relevance based on persona and job"""
#         text = text.lower()
#         doc = nlp(text)
        
#         # Count keyword matches
#         keyword_matches = sum(
#             1 for token in doc 
#             if token.lemma_.lower() in self.keywords and not token.is_stop
#         )
        
#         # Calculate density score
#         word_count = max(1, len([t for t in doc if not t.is_punct]))
#         density_score = keyword_matches / word_count
        
#         return density_score * 100  # Scale to percentage

# class PDFStructureExtractor:
#     def __init__(self):
#         self.font_stats = defaultdict(list)
#         self.median_size = 12  # Default fallback

#     def analyze_fonts(self, page):
#         """Collect font statistics without bold detection"""
#         words = page.extract_words(extra_attrs=["size", "fontname"])
#         for word in words:
#             if word["text"].strip():
#                 self.font_stats[word["fontname"]].append(word["size"])
#         return words

#     def compute_typography(self):
#         """Calculate dominant font characteristics"""
#         all_sizes = []
#         for sizes in self.font_stats.values():
#             all_sizes.extend(sizes)
#         if all_sizes:
#             self.median_size = np.median(all_sizes)

#     def detect_heading_level(self, text: str, font_size: float) -> Optional[str]:
#         """Reliable heading detection using multiple heuristics"""
#         clean_txt = clean_text(text)
        
#         # Skip obvious non-headings
#         if len(clean_txt) > 100 or len(clean_txt.split()) > 8:
#             return None
        
#         # Size-based detection
#         size_ratio = font_size / self.median_size if self.median_size else 1
        
#         # Content patterns (ordered by specificity)
#         patterns = [
#             (r'^(?:APPENDIX|Appendix|CHAPTER|Chapter|SECTION|Section)\b', "H1"),
#             (r'^\d+\.\d+\.\d+\s', "H3"),
#             (r'^\d+\.\d+\s', "H2"),
#             (r'^\d+\.\s', "H2"),
#             (r'^[IVXLCDM]+\.', "H1"),
#             (r'^[A-Z][A-Z0-9\s]+:$', "H2"),
#             (r'^[A-Z][a-z]+:', "H3"),
#             (r'^•\s', "H4")
#         ]
        
#         for pattern, level in patterns:
#             if re.match(pattern, clean_txt):
#                 return level
        
#         # Size heuristics (more conservative)
#         if size_ratio > 2.0:
#             return "H1" if len(clean_txt.split()) <= 5 else "H2"
#         elif size_ratio > 1.5:
#             return "H2" if clean_txt.endswith(':') else "H3"
        
#         return None

# def extract_pdf_content(pdf_path: Path, analyzer: PersonaAnalyzer) -> Dict:
#     """Extract and score content based on persona relevance"""
#     extractor = PDFStructureExtractor()
#     result = {"title": "", "sections": []}
    
#     try:
#         with pdfplumber.open(pdf_path) as pdf:
#             # First pass: document analysis
#             for page in pdf.pages:
#                 extractor.analyze_fonts(page)
#             extractor.compute_typography()
            
#             # Second pass: content extraction and scoring
#             for page_num, page in enumerate(pdf.pages):
#                 text = page.extract_text()
#                 if not text:
#                     continue
                
#                 # Score entire page
#                 page_score = analyzer.score_relevance(text)
                
#                 # Extract sections
#                 words = page.extract_words(extra_attrs=["size", "fontname"])
#                 current_line = ""
#                 current_size = None
                
#                 for word in words:
#                     word_text = clean_text(word["text"])
#                     if not word_text:
#                         continue
                    
#                     # New line on significant size change
#                     if current_size and abs(word["size"] - current_size) > 2.0:
#                         if current_line:
#                             level = extractor.detect_heading_level(current_line, current_size)
#                             if level:
#                                 section_score = analyzer.score_relevance(current_line)
#                                 result["sections"].append({
#                                     "text": current_line,
#                                     "page": page_num + 1,  # 1-indexed for output
#                                     "level": level,
#                                     "score": section_score,
#                                     "page_score": page_score
#                                 })
#                         current_line = word_text
#                         current_size = word["size"]
#                     else:
#                         current_line += (" " + word_text) if current_line else word_text
#                         current_size = current_size or word["size"]
                
#                 # Process remaining content
#                 if current_line:
#                     level = extractor.detect_heading_level(current_line, current_size)
#                     if level:
#                         section_score = analyzer.score_relevance(current_line)
#                         result["sections"].append({
#                             "text": current_line,
#                             "page": page_num + 1,
#                             "level": level,
#                             "score": section_score,
#                             "page_score": page_score
#                         })
            
#             # Determine title
#             if not result["title"]:
#                 first_page = pdf.pages[0].extract_text()
#                 if first_page:
#                     result["title"] = clean_text(first_page.split('\n')[0])
    
#     except Exception as e:
#         print(f"Error processing {pdf_path.name}: {str(e)}")
#         return {"title": f"Error: {str(e)}", "sections": []}
    
#     return result

# def process_collection(input_path: Path) -> Dict:
#     """Process a complete document collection"""
#     # Load configuration
#     with open(input_path / "challenge1b_input.json") as f:
#         config = Config(**json.load(f))
    
#     # Initialize persona analyzer
#     analyzer = PersonaAnalyzer(config.persona["role"], config.job_to_be_done["task"])
    
#     results = []
#     pdf_dir = input_path / "PDFs"
    
#     # Process each document
#     for doc in config.documents:
#         pdf_path = pdf_dir / doc["filename"]
#         if pdf_path.exists():
#             content = extract_pdf_content(pdf_path, analyzer)
#             content["filename"] = doc["filename"]
#             results.append(content)
    
#     # Rank sections across all documents
#     all_sections = []
#     for doc in results:
#         for section in doc["sections"]:
#             # Combined score (section relevance + page relevance)
#             combined_score = (section["score"] * 0.7) + (section["page_score"] * 0.3)
#             all_sections.append({
#                 "document": doc["filename"],
#                 "page_number": section["page"],
#                 "section_title": section["text"],
#                 "score": combined_score,
#                 "level": section["level"]
#             })
    
#     # Sort by score and level importance
#     all_sections.sort(key=lambda x: (-x["score"], -len(x["level"])))
    
#     # Prepare output
#     output = {
#         "metadata": {
#             "input_documents": [doc["filename"] for doc in config.documents],
#             "persona": config.persona["role"],
#             "job_to_be_done": config.job_to_be_done["task"],
#             "processing_timestamp": datetime.now().isoformat()
#         },
#         "extracted_sections": [],
#         "subsection_analysis": []
#     }
    
#     # Select top sections (max 10)
#     top_sections = all_sections[:10]
#     for i, section in enumerate(top_sections, 1):
#         output["extracted_sections"].append({
#             "document": section["document"],
#             "page_number": section["page_number"],
#             "section_title": section["section_title"],
#             "importance_rank": i
#         })
    
#     # Add subsection analysis (top 3 sections expanded)
#     for section in top_sections[:3]:
#         doc_path = pdf_dir / section["document"]
#         try:
#             with pdfplumber.open(doc_path) as pdf:
#                 page = pdf.pages[section["page_number"] - 1]
#                 text = page.extract_text()
#                 if text:
#                     # Extract text around the section
#                     lines = text.split('\n')
#                     section_idx = next(
#                         (i for i, line in enumerate(lines) 
#                          if section["section_title"] in line), 0
#                     )
#                     start = max(0, section_idx - 2)
#                     end = min(len(lines), section_idx + 5)
#                     refined_text = '\n'.join(lines[start:end])
                    
#                     output["subsection_analysis"].append({
#                         "document": section["document"],
#                         "page_number": section["page_number"],
#                         "refined_text": refined_text,
#                         "constraints": [
#                             f"Character limit: {len(refined_text)}",
#                             f"Relevance score: {section['score']:.1f}"
#                         ]
#                     })
#         except Exception as e:
#             print(f"Error extracting subsection from {section['document']}: {str(e)}")
    
#     return output

# def main():
#     print("=== Starting Persona-Driven Document Analyzer ===")
#     print(f"Current directory: {Path.cwd()}")
#     print(f"Input directory exists: {Path('./input').exists()}")
#     print(f"Collections found: {list(Path('./input').glob('Collection_*'))}")
#     input_dir = Path("./input")
#     output_dir = Path("./output")
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     # Process each collection
#     for collection_dir in input_dir.glob("Collection_*"):
#         print(f"Processing {collection_dir.name}...")
#         result = process_collection(collection_dir)
        
#         # Save results
#         output_file = output_dir / f"{collection_dir.name}_output.json"
#         with open(output_file, "w") as f:
#             json.dump(Output(**result).dict(), f, indent=2)
        
#         print(f"Saved results to {output_file}")

# if __name__ == "__main__":
#     print("=== Starting Persona-Driven Document Analyzer ===")
#     main()



# import json
# import re
# from pathlib import Path
# from typing import List, Dict, Optional, Any
# from pydantic import BaseModel
# from datetime import datetime
# import spacy
# import numpy as np
# from collections import defaultdict

# # Initialize NLP
# nlp = spacy.load("en_core_web_sm")

# # Configuration to point to Round 1A output
# ROUND_1A_OUTPUT = Path(r"C:\Users\kkrit\OneDrive\Desktop\Adobe Hackathon\Round_1A\output")

# class Config(BaseModel):
#     challenge_info: Dict[str, str]
#     documents: List[Dict[str, str]]
#     persona: Dict[str, str]
#     job_to_be_done: Dict[str, str]

# class Section(BaseModel):
#     document: str
#     page_number: int
#     section_title: str
#     importance_rank: int

# class SubSection(BaseModel):
#     document: str
#     page_number: int
#     refined_text: str
#     constraints: List[str]

# class Output(BaseModel):
#     metadata: Dict[str, Any]
#     extracted_sections: List[Section]
#     subsection_analysis: List[SubSection]

# def clean_text(text: str) -> str:
#     """Advanced text normalization that preserves structure"""
#     text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)  # Remove control chars
#     text = re.sub(r'(?<!\n)\s+', ' ', text)  # Normalize spaces
#     return text.strip()

# class PersonaAnalyzer:
#     def __init__(self, persona: str, job: str):
#         self.persona = persona.lower()
#         self.job = job.lower()
#         self.keywords = self._extract_keywords()
        
#         # Add persona-specific boosts
#         if "travel" in self.persona:
#             self.keywords.extend([
#                 "itinerary", "accommodation", "transport",
#                 "attraction", "guide", "tour", "plan",
#                 "hotel", "restaurant", "activity", "itinerary",
#                 "sightseeing", "recommendation", "destination"
#             ])
    
#     def _extract_keywords(self) -> List[str]:
#         """Extract relevant keywords from persona and job description"""
#         doc = nlp(f"{self.persona} {self.job}")
#         keywords = []
        
#         # Extract nouns and verbs
#         for token in doc:
#             if token.pos_ in ["NOUN", "PROPN", "VERB"] and not token.is_stop:
#                 keywords.append(token.lemma_.lower())
        
#         return list(set(keywords))
    
#     def score_relevance(self, text: str) -> float:
#         """Score text relevance based on persona and job"""
#         text = text.lower()
#         doc = nlp(text)
        
#         # Count keyword matches
#         keyword_matches = sum(
#             1 for token in doc 
#             if token.lemma_.lower() in self.keywords and not token.is_stop
#         )
        
#         # Calculate density score
#         word_count = max(1, len([t for t in doc if not t.is_punct]))
#         density_score = keyword_matches / word_count
        
#         # Boost scores for certain patterns
#         if any(kw in text for kw in ["itinerary", "recommendation", "guide"]):
#             density_score *= 1.5
#         elif any(kw in text for kw in ["hotel", "restaurant", "activity"]):
#             density_score *= 1.3
            
#         return min(density_score * 100, 100)  # Scale to percentage, cap at 100

# def load_extracted_content(pdf_filename: str) -> Dict:
#     """Load pre-extracted content from Round 1A output"""
#     json_path = ROUND_1A_OUTPUT / f"{Path(pdf_filename).stem}.json"
#     try:
#         with open(json_path) as f:
#             data = json.load(f)
#             return {
#                 "title": data.get("title", ""),
#                 "sections": [
#                     {
#                         "text": item["text"],
#                         "page": item["page"] + 1,  # Convert to 1-based
#                         "level": item["level"],
#                         "content": item.get("content", "")  # If you stored content
#                     }
#                     for item in data.get("outline", [])
#                 ]
#             }
#     except Exception as e:
#         print(f"Error loading {json_path}: {str(e)}")
#         return {"title": "", "sections": []}

# def process_collection(input_path: Path) -> Dict:
#     """Process a complete document collection"""
#     # Load configuration
#     with open(input_path / "challenge1b_input.json") as f:
#         config_data = json.load(f)
#         config = Config(**config_data)
    
#     # Initialize persona analyzer
#     analyzer = PersonaAnalyzer(
#         config.persona["role"],
#         config.job_to_be_done["task"]
#     )
    
#     # Process each document using pre-extracted JSON
#     all_sections = []
#     for doc in config.documents:
#         content = load_extracted_content(doc["filename"])
        
#         for section in content["sections"]:
#             score = analyzer.score_relevance(section["text"])
#             all_sections.append({
#                 "document": doc["filename"],
#                 "page_number": section["page"],
#                 "section_title": section["text"],
#                 "score": score,
#                 "level": section["level"],
#                 "content": section.get("content", "")
#             })
    
#     # Sort by score (descending) and level importance
#     all_sections.sort(key=lambda x: (-x["score"], x["level"]))
    
#     # Prepare output
#     output = {
#         "metadata": {
#             "input_documents": [doc["filename"] for doc in config.documents],
#             "persona": config.persona["role"],
#             "job_to_be_done": config.job_to_be_done["task"],
#             "processing_timestamp": datetime.now().isoformat()
#         },
#         "extracted_sections": [],
#         "subsection_analysis": []
#     }
    
#     # Select top sections (max 10)
#     top_sections = all_sections[:10]
#     for rank, section in enumerate(top_sections, 1):
#         output["extracted_sections"].append({
#             "document": section["document"],
#             "page_number": section["page_number"],
#             "section_title": section["section_title"],
#             "importance_rank": rank
#         })
        
#         # For top 3 sections, include content
#         if rank <= 3:
#             output["subsection_analysis"].append({
#                 "document": section["document"],
#                 "page_number": section["page_number"],
#                 "refined_text": section["content"] or section["section_title"],
#                 "constraints": [
#                     f"Relevance score: {section['score']:.1f}",
#                     f"Level: {section['level']}"
#                 ]
#             })
    
#     return output

# def main():
#     print("=== Starting Persona-Driven Document Analyzer ===")
#     print(f"Current directory: {Path.cwd()}")
#     print(f"Input directory exists: {Path('./input').exists()}")
#     print(f"Collections found: {list(Path('./input').glob('Collection_*'))}")
    
#     input_dir = Path("./input")
#     output_dir = Path("./output")
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     # Process each collection
#     for collection_dir in input_dir.glob("Collection_*"):
#         print(f"Processing {collection_dir.name}...")
#         result = process_collection(collection_dir)
        
#         # Save results
#         output_file = output_dir / f"{collection_dir.name}_output.json"
#         with open(output_file, "w") as f:
#             json.dump(Output(**result).dict(), f, indent=2)
        
#         print(f"Saved results to {output_file}")

# if __name__ == "__main__":
#     main()

import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Any
from pydantic import BaseModel
from datetime import datetime
import spacy
from collections import defaultdict

# Initialize NLP
nlp = spacy.load("en_core_web_sm")

class Config(BaseModel):
    challenge_info: Dict[str, str]
    documents: List[Dict[str, str]]
    persona: Dict[str, str]
    job_to_be_done: Dict[str, str]

class Section(BaseModel):
    document: str
    page_number: int
    section_title: str
    importance_rank: int

class SubSection(BaseModel):
    document: str
    page_number: int
    refined_text: str
    constraints: List[str]

class Output(BaseModel):
    metadata: Dict[str, Any]
    extracted_sections: List[Section]
    subsection_analysis: List[SubSection]

def clean_text(text: str) -> str:
    """Advanced text normalization that preserves structure"""
    text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)  # Remove control chars
    text = re.sub(r'(?<!\n)\s+', ' ', text)  # Normalize spaces
    return text.strip()

class PersonaAnalyzer:
    def __init__(self, persona: str, job: str):
        self.persona = persona.lower()
        self.job = job.lower()
        self.keywords = self._extract_keywords()
        
        # Add persona-specific boosts
        if "travel" in self.persona:
            self.keywords.extend([
                "itinerary", "accommodation", "transport",
                "attraction", "guide", "tour", "plan",
                "hotel", "restaurant", "activity", "itinerary",
                "sightseeing", "recommendation", "destination"
            ])
        elif "hr" in self.persona or "professional" in self.persona:
            self.keywords.extend(["form", "onboarding", "compliance", "document"])
        elif "food" in self.persona or "contractor" in self.persona:
            self.keywords.extend(["recipe", "ingredient", "menu", "vegetarian", "buffet"])
    
    def _extract_keywords(self) -> List[str]:
        """Extract relevant keywords from persona and job description"""
        doc = nlp(f"{self.persona} {self.job}")
        keywords = []
        
        # Extract nouns and verbs
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN", "VERB"] and not token.is_stop:
                keywords.append(token.lemma_.lower())
        
        return list(set(keywords))
    
    def score_relevance(self, text: str) -> float:
        """Score text relevance based on persona and job"""
        text = text.lower()
        doc = nlp(text)
        
        # Count keyword matches
        keyword_matches = sum(
            1 for token in doc 
            if token.lemma_.lower() in self.keywords and not token.is_stop
        )
        
        # Calculate density score
        word_count = max(1, len([t for t in doc if not t.is_punct]))
        density_score = keyword_matches / word_count
        
        # Boost scores for certain patterns
        if any(kw in text for kw in ["itinerary", "recommendation", "guide"]):
            density_score *= 1.5
        elif any(kw in text for kw in ["hotel", "restaurant", "activity"]):
            density_score *= 1.3
            
        return min(density_score * 100, 100)  # Scale to percentage, cap at 100

def load_extracted_content(collection_path: Path, pdf_filename: str) -> Dict:
    """Load pre-extracted content from Processed_pdf folder"""
    json_path = collection_path / "Processed_pdf" / f"{Path(pdf_filename).stem}.json"
    try:
        with open(json_path) as f:
            data = json.load(f)
            sections = []
            
            # Handle different JSON structures
            if "outline" in data:  # Round 1A format
                sections = [
                    {
                        "text": item["text"],
                        "page": item["page"] + 1,  # Convert to 1-based
                        "level": item["level"],
                        "content": item.get("content", item["text"])
                    }
                    for item in data.get("outline", [])
                ]
            elif "sections" in data:  # Alternative format
                sections = [
                    {
                        "text": item["title"],
                        "page": item.get("page", 1),
                        "level": item.get("level", "H2"),
                        "content": item.get("content", item["title"])
                    }
                    for item in data.get("sections", [])
                ]
            
            return {
                "title": data.get("title", Path(pdf_filename).stem),
                "sections": sections
            }
    except Exception as e:
        print(f"Error loading {json_path}: {str(e)}")
        return {"title": Path(pdf_filename).stem, "sections": []}

def process_collection(collection_path: Path) -> Dict:
    """Process a complete document collection"""
    # Load configuration
    with open(collection_path / "challenge1b_input.json") as f:
        config_data = json.load(f)
        config = Config(**config_data)
    
    # Initialize persona analyzer
    analyzer = PersonaAnalyzer(
        config.persona["role"],
        config.job_to_be_done["task"]
    )
    
    # Process each document using pre-extracted JSON
    all_sections = []
    for doc in config.documents:
        content = load_extracted_content(collection_path, doc["filename"])
        
        for section in content["sections"]:
            score = analyzer.score_relevance(section["text"])
            all_sections.append({
                "document": doc["filename"],
                "page_number": section["page"],
                "section_title": section["text"],
                "score": score,
                "level": section["level"],
                "content": section.get("content", "")
            })
    
    # Sort by score (descending) and level importance
    all_sections.sort(key=lambda x: (-x["score"], x["level"]))
    
    # Prepare output
    output = {
        "metadata": {
            "input_documents": [doc["filename"] for doc in config.documents],
            "persona": config.persona["role"],
            "job_to_be_done": config.job_to_be_done["task"],
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": [],
        "subsection_analysis": []
    }
    
    # Select top sections (max 10)
    top_sections = all_sections[:10]
    for rank, section in enumerate(top_sections, 1):
        output["extracted_sections"].append({
            "document": section["document"],
            "page_number": section["page_number"],
            "section_title": section["section_title"],
            "importance_rank": rank
        })
        
        # For top 3 sections, include content
        if rank <= 3:
            output["subsection_analysis"].append({
                "document": section["document"],
                "page_number": section["page_number"],
                "refined_text": section["content"] or section["section_title"],
                "constraints": [
                    f"Relevance score: {section['score']:.1f}",
                    f"Level: {section['level']}"
                ]
            })
    
    return output

def main():
    print("=== Starting Persona-Driven Document Analyzer ===")
    print(f"Current directory: {Path.cwd()}")
    print(f"Input directory exists: {Path('./input').exists()}")
    
    input_dir = Path("./input")
    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each collection
    for collection_dir in input_dir.glob("Collection_*"):
        print(f"\nProcessing {collection_dir.name}...")
        print(f"Found {len(list((collection_dir / 'Processed_pdf').glob('*.json')))} JSON files in Processed_pdf")
        
        result = process_collection(collection_dir)
        
        # Save results
        output_file = output_dir / f"{collection_dir.name}_output.json"
        with open(output_file, "w") as f:
            json.dump(Output(**result).dict(), f, indent=2)
        
        print(f"Saved results to {output_file}")
        print(f"Extracted {len(result['extracted_sections'])} sections")
        print(f"Generated {len(result['subsection_analysis'])} subsection analyses")

if __name__ == "__main__":
    main()