import io
import os
import zipfile
import streamlit as st
from app_pipeline import DocumentExtractor, AnonymizationEngine
from vector_store import VectorSearchEngine

st.set_page_config(page_title="Privacy-Preserving Resume RAG", layout="wide")
st.title("🔒 Privacy-Preserving Resume Parser & Semantic Search")

@st.cache_resource
def load_engines():
    return AnonymizationEngine(), VectorSearchEngine()

anonymizer, vector_engine = load_engines()

def process_single_pdf(pdf_bytes, candidate_id):
    raw_text = DocumentExtractor.extract_text_from_pdf(pdf_bytes)
    if raw_text.strip():
        redacted_text = anonymizer.anonymize_cv(raw_text)
        vector_engine.upsert_cv(candidate_id=candidate_id, redacted_text=redacted_text)
        return True
    return False

# Sidebar - Document Ingestion (Supports PDF & ZIP)
st.sidebar.header("Upload Resumes")
uploaded_files = st.sidebar.file_uploader(
    "Upload Resumes (PDF or ZIP archives)", 
    type=["pdf", "zip"], 
    accept_multiple_files=True
)

if uploaded_files:
    if st.sidebar.button("Process & Index All Uploads"):
        with st.sidebar.spinner("Processing documents in memory..."):
            total_indexed = 0
            
            for file_item in uploaded_files:
                # 1. Handle ZIP Archives (Batch Uploads)
                if file_item.name.lower().endswith(".zip"):
                    try:
                        with zipfile.ZipFile(io.BytesIO(file_item.read())) as z:
                            for inner_filename in z.namelist():
                                # Ignore hidden system metadata files
                                if (
                                    inner_filename.lower().endswith(".pdf") 
                                    and not inner_filename.startswith("__MACOSX")
                                    and not os.path.basename(inner_filename).startswith("._")
                                ):
                                    pdf_bytes = z.read(inner_filename)
                                    clean_name = os.path.basename(inner_filename)
                                    if process_single_pdf(pdf_bytes, candidate_id=clean_name):
                                        total_indexed += 1
                                        st.sidebar.caption(f"✓ Indexed: {clean_name}")
                    except Exception as e:
                        st.sidebar.error(f"Failed to process archive {file_item.name}: {e}")

                # 2. Handle Individual PDF Files
                elif file_item.name.lower().endswith(".pdf"):
                    pdf_bytes = file_item.read()
                    if process_single_pdf(pdf_bytes, candidate_id=file_item.name):
                        total_indexed += 1
                        st.sidebar.caption(f"✓ Indexed: {file_item.name}")
            
            st.sidebar.success(f"Successfully indexed {total_indexed} candidate(s)!")

# Main View - Search Interface
st.header("Candidate Semantic Search")
query = st.text_input(
    "Enter Job Description or Search Criteria:", 
    "Senior Accountant with GAAP compliance, financial reporting, and general ledger reconciliation experience"
)

col1, col2 = st.columns([1, 4])
with col1:
    top_k = st.number_input("Top Matches", min_value=1, max_value=10, value=3)

if st.button("Search Candidates"):
    if query.strip():
        with st.spinner("Searching index..."):
            results = vector_engine.search_candidates(query=query, top_k=top_k)
            
            st.subheader("Top Matches")
            if hasattr(results, 'matches') and results.matches:
                for idx, match in enumerate(results.matches):
                    score_pct = match.score * 100
                    with st.expander(f"Candidate ID: {match.id} (Match Confidence: {score_pct:.1f}%)"):
                        st.text_area(
                            "Sanitized Resume Content", 
                            match.metadata.get("sanitized_text", ""), 
                            height=200,
                            key=f"text_area_{match.id}_{idx}"  # Unique key prevents StreamlitDuplicateElementId error
                        )
            else:
                st.info("No matching candidates found.")
    else:
        st.warning("Please enter a query before searching.")