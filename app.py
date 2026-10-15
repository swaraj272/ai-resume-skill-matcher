import re
from pathlib import Path
import streamlit as st
import fitz  # PyMuPDF
from docx import Document
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Skill Matcher", page_icon="🧠", layout="wide")

SKILLS = [
    "python", "java", "c++", "sql", "machine learning", "deep learning", "nlp",
    "natural language processing", "pandas", "numpy", "scikit-learn", "tensorflow",
    "pytorch", "data analysis", "data visualization", "power bi", "excel", "statistics",
    "feature engineering", "classification", "regression", "clustering", "git", "github",
    "rest api", "flask", "fastapi", "streamlit", "langchain", "rag", "embeddings",
    "vector database", "faiss", "llm", "large language models", "prompt engineering",
    "aws", "docker", "linux", "html", "css", "javascript", "mysql", "mongodb"
]

@st.cache_resource
def load_model():
    # First run downloads this pretrained sentence-embedding model.
    return SentenceTransformer("all-MiniLM-L6-v2")

def extract_text(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    data = uploaded_file.getvalue()
    if suffix == ".pdf":
        with fitz.open(stream=data, filetype="pdf") as pdf:
            return "\n".join(page.get_text() for page in pdf)
    if suffix == ".docx":
        import io
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)
    if suffix == ".txt":
        return data.decode("utf-8", errors="ignore")
    raise ValueError("Please upload a PDF, DOCX, or TXT file.")

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#.\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def find_skills(text):
    normalized = normalize(text)
    found = []
    for skill in SKILLS:
        # Match whole phrases rather than partial words where possible.
        pattern = r"(?<![a-z0-9])" + re.escape(skill) + r"(?![a-z0-9])"
        if re.search(pattern, normalized):
            found.append(skill)
    return sorted(set(found))

st.title("🧠 AI Resume Skill Matcher")
st.write("Compare a resume with a job description using semantic embeddings and skill extraction.")
st.info("Privacy note: this demo processes uploaded files in the running app. Do not upload sensitive documents to a public deployment without reviewing its privacy and storage settings.")

left, right = st.columns(2)
with left:
    resume_file = st.file_uploader("Upload resume (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
with right:
    jd = st.text_area("Paste job description", height=230, placeholder="Paste the complete job description here...")

if st.button("Analyze match", type="primary", use_container_width=True):
    if not resume_file or not jd.strip():
        st.error("Upload a resume and paste a job description first.")
    else:
        try:
            resume_text = extract_text(resume_file)
            if len(resume_text.strip()) < 30:
                st.error("Very little text was extracted. If this is a scanned PDF, run OCR first or upload a text-based PDF/DOCX/TXT.")
            else:
                with st.spinner("Loading the embedding model and comparing text..."):
                    model = load_model()
                    vectors = model.encode([resume_text, jd], normalize_embeddings=True)
                    score = float(cosine_similarity([vectors[0]], [vectors[1]])[0][0])
                    score_pct = max(0.0, min(100.0, score * 100))
                    resume_skills = set(find_skills(resume_text))
                    jd_skills = set(find_skills(jd))
                    matched = sorted(resume_skills & jd_skills)
                    missing = sorted(jd_skills - resume_skills)
                st.subheader("Match report")
                c1, c2, c3 = st.columns(3)
                c1.metric("Semantic similarity", f"{score_pct:.1f}%")
                c2.metric("Matched listed skills", len(matched))
                c3.metric("Potential missing skills", len(missing))
                st.progress(score_pct / 100)
                st.caption("Similarity is a text-comparison signal, not a probability of getting hired. It may be influenced by resume length and wording.")
                a, b = st.columns(2)
                with a:
                    st.markdown("#### Skills found in both")
                    st.write(", ".join(matched) if matched else "No listed skills matched.")
                with b:
                    st.markdown("#### Skills in job description not found in resume")
                    st.write(", ".join(missing) if missing else "No missing skills from this demo dictionary.")
                st.markdown("#### Skills detected in resume")
                st.write(", ".join(sorted(resume_skills)) if resume_skills else "No skills from the built-in dictionary detected.")
                st.markdown("#### Skills detected in job description")
                st.write(", ".join(sorted(jd_skills)) if jd_skills else "No skills from the built-in dictionary detected.")
                st.download_button("Download text report", data=(
                    f"AI Resume Skill Matcher Report\nSemantic similarity: {score_pct:.1f}%\n"
                    f"Matched skills: {', '.join(matched)}\nPotential missing skills: {', '.join(missing)}\n"
                ), file_name="resume_match_report.txt", mime="text/plain")
        except Exception as exc:
            st.error(f"Could not analyze these files: {exc}")
            st.caption("Check that the file is a valid PDF/DOCX/TXT and that the model downloaded successfully.")

with st.expander("How this project works"):
    st.markdown("""
    1. **Document parsing:** PyMuPDF or python-docx extracts text from the resume.
    2. **Preprocessing:** basic normalization makes text easier to inspect.
    3. **Skill extraction:** a small transparent dictionary detects selected skill phrases.
    4. **Embeddings:** `all-MiniLM-L6-v2` maps resume and job text to dense vectors.
    5. **Cosine similarity:** compares vector direction to estimate semantic similarity.
    6. **Report:** displays similarity, overlap, and skills mentioned in the job description but not detected in the resume.

    Limitations: the skill dictionary is not exhaustive, and missing means “not detected,” not proof that the candidate lacks the skill. This is a decision-support demo, not an automated hiring decision system.
    """)
