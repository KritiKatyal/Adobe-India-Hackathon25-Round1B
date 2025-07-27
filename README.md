# Challenge 1b: Multi-Collection PDF Analysis


### Team CodeHers  
**Kriti Katyal**  
**Siya Verma**

---

## Overview
An intelligent document analysis solution that extracts and prioritizes relevant content from PDF collections based on specific personas and their tasks. The system combines PDF structure analysis with NLP techniques to identify the most valuable sections for each user scenario.
---

## System Architecture
```
Challenge_1b/
├── input/
│   ├── Collection_1/               # Document collection 1
│   │   ├── PDFs/                   # Source PDF files
│   │   ├── challenge1b_input.json  # Persona/task configuration
│   │   └── Processed_pdf/          # Processed pdfs (through Challenge_1A logic)
│   ├── Collection_2/
│   │   ├── PDFs/
│   │   ├── challenge1b_input.json  # Persona/task configuration
│   │   └── Processed_pdf/          # Processed pdfs (through Challenge_1A logic)
│   └── ...                        # Additional collections
├── output/
│   ├── Collection_1_output.json    # Final analysis results
│   ├── Collection_2_output.json    # Final analysis results
│   └── Collection_3_output.json    # Final analysis results
├── src/
│   ├── process_pdf.py             # PDF processing logic
│   └── main.py                    # Core orchestration logic
├── Dockerfile
└── README.md
```

---
## Technical Approach

### PDF Processing Pipeline
1. **Structure Extraction** (process_pdf.py):
   - Analyzes font characteristics to identify headings
   - Detects hierarchical document structure (H1-H4)
   - Processes each PDF into structured JSON format

2. **Persona Analysis** (main.py):
   - Creates keyword profiles based on role and task descriptions
   - Scores content relevance using NLP techniques
   - Ranks sections by importance to the specified job-to-be-done

3. **Output Generation**:
   - Creates structured JSON output with:
     - Document metadata
     - Ranked extracted sections
     - Detailed subsection analyses


## Installation & Execution

### Prerequisites
- Docker (recommended)
- Python 3.9+ (for local development)

### Running with Docker
```bash
# Build the Docker image
docker build -t persona-doc-intel .

# Run the container
docker run -v "/path/to/input:/app/input" -v "/path/to/output:/app/output" persona-doc-intel
```

#### Example (Windows paths)
```bash
# Build the Docker image
docker build -t persona-doc-intel .

# Run the container
docker run -v "C:\Users\kkrit\OneDrive\Desktop\Adobe-India-Hackathon25\Challenge_1b\input:/app/input" -v "C:\Users\kkrit\OneDrive\Desktop\Adobe-India-Hackathon25\Challenge_1b\output:/app/output" persona-doc-intel
```

---


## Key Features
- **Persona-aware content extraction**: Tailors results to different user roles and needs  
- **Multi-document analysis**: Processes 3–10 related PDFs simultaneously  
- **Intelligent ranking**: Scores sections by relevance to the specified job-to-be-done  
- **Structured output**: Provides clear metadata, ranked sections, and detailed analyses  
- **Efficient processing**: Runs on CPU-only with fast execution times  

## Technical Approach
1. **PDF Structure Analysis**: Uses font characteristics and layout patterns to identify headings and sections  
2. **Content Extraction**: Captures both section titles and their associated content  
3. **Persona Modeling**: Creates keyword profiles based on role and task descriptions  
4. **Relevance Scoring**: Computes importance scores using NLP techniques  
5. **Result Compilation**: Generates structured JSON output with ranked sections  
<!-- 


Example-
docker build -t persona-doc-intel .     
docker run -v "C:\Users\kkrit\OneDrive\Desktop\Adobe Hackathon\Challenge_1b\input:/app/input" -v "C:\Users\kkrit\OneDrive\Desktop\Adobe Hackathon\Challenge_1b\output:/app/output" persona-doc-intel -->