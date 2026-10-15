# AI Resume Skill Matcher

A beginner-friendly AI/ML portfolio project that compares a resume to a job description using sentence embeddings, cosine similarity, and a transparent skill dictionary.

## Run locally

1. Install Python 3.10 or 3.11.
2. Open a terminal in this folder.
3. Create and activate a virtual environment:
   - macOS/Linux: `python3 -m venv .venv && source .venv/bin/activate`
   - Windows: `py -m venv .venv && .venv\\Scripts\\activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run: `streamlit run app.py`
6. The first run downloads `all-MiniLM-L6-v2`; an internet connection is required for that initial download.

## Test it

Use a text-based sample resume PDF/DOCX/TXT and paste a job description containing a few skills such as Python, SQL, NLP, and machine learning. Compare the report with the source text.

## Core AI/ML concepts

- NLP and document text extraction
- Sentence embeddings
- Semantic similarity
- Cosine similarity
- Rule-based skill extraction
- Evaluation and limitations

## Honest project description for a resume (after you build and test it)

Built a Streamlit application that extracts text from resumes, compares it with job descriptions using sentence-transformer embeddings and cosine similarity, and reports overlapping and potentially missing skills.

## Improvements to try

- Add a larger, carefully maintained skills taxonomy and synonyms.
- Add unit tests for file parsing and skill matching.
- Compare multiple job descriptions and rank them.
- Evaluate with a small set of manually labelled resume/job pairs.
- Add OCR for scanned PDFs only after the basic version works.
