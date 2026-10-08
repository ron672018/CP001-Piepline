import io
import re
import fitz  # PyMuPDF
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from vector_store import VectorSearchEngine


# ==========================================
# MODULE 1: Document Extraction Module
# ==========================================
class DocumentExtractor:
    @staticmethod
    def extract_text_from_pdf(pdf_bytes: bytes) -> str:
        """Extracts text safely from a PDF byte stream using PyMuPDF."""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            extracted_text = []
            for page in doc:
                text = page.get_text("text")
                if text:
                    extracted_text.append(text)
            doc.close()
            return "\n".join(extracted_text)
        except Exception as e:
            print(f"Error extracting PDF text: {str(e)}")
            return ""


# ==========================================
# MODULE 2: Hybrid Anonymization Engine
# ==========================================
class AnonymizationEngine:
    def __init__(self):
        self.analyzer = AnalyzerEngine()
        self.anonymizer = AnonymizerEngine()

    def _fast_regex_pass(self, text: str) -> str:
        """Microsecond Regex pass for deterministic PII (Emails, Phones, URLs)."""
        # Redact Emails
        text = re.sub(
            r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            '[REDACTED_EMAIL]',
            text
        )
        # Redact Phone Numbers (international & standard formats)
        text = re.sub(
            r'(\+?\d{1,3}[\s-]?)?\(?\d{2,4}\)?[\s-]?\d{3,4}[\s-]?\d{3,4}',
            '[REDACTED_PHONE]',
            text
        )
        # Redact URLs (LinkedIn, GitHub, Portfolios)
        text = re.sub(
            r'https?://[^\s]+|www\.[^\s]+|linkedin\.com/in/[^\s]+',
            '[REDACTED_URL]',
            text
        )
        return text

    def anonymize_cv(self, raw_text: str) -> str:
        """Runs hybrid pass: Fast Regex -> Contextual Presidio NER."""
        if not raw_text.strip():
            return ""

        pre_processed = self._fast_regex_pass(raw_text)

        results = self.analyzer.analyze(
            text=pre_processed,
            entities=["PERSON", "LOCATION", "ORGANIZATION"],
            language="en"
        )

        anonymized_result = self.anonymizer.anonymize(
            text=pre_processed,
            analyzer_results=results,
            operators={
                "PERSON": OperatorConfig("replace", {"new_value": "[REDACTED_NAME]"}),
                "LOCATION": OperatorConfig("replace", {"new_value": "[REDACTED_LOCATION]"}),
                "ORGANIZATION": OperatorConfig("replace", {"new_value": "[REDACTED_ORGANIZATION]"}),
            }
        )
        return anonymized_result.text


# ==========================================
# HELPER: Generate Sample PDF Resumes
# ==========================================
def create_sample_pdf(text: str) -> bytes:
    """Helper to create a temporary PDF resume in memory."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# ==========================================
# MAIN PIPELINE EXECUTION
# ==========================================
if __name__ == "__main__":
    candidates_sample_data = {
        "candidate_001": (
            "John Doe\nEmail: john@example.com | Phone: 555-019-2834\nLocation: Seattle, WA\n"
            "Senior Cloud Engineer at Amazon. 6 years experience with AWS, Kubernetes, Terraform, and Python."
        ),
        "candidate_002": (
            "Jane Smith\nEmail: jane@example.com | Phone: 555-018-9988\nLocation: New York, NY\n"
            "Data Scientist at Google. Specialized in Machine Learning, NLP, PyTorch, and Python data pipelines."
        )
    }

    print("--- 1. Initializing Modules & Vector Search Engine ---")
    anonymizer = AnonymizationEngine()
    vector_engine = VectorSearchEngine()

    print("\n--- 2. Processing & Indexing Candidates ---")
    for cand_id, raw_resume in candidates_sample_data.items():
        pdf_bytes = create_sample_pdf(raw_resume)
        extracted_text = DocumentExtractor.extract_text_from_pdf(pdf_bytes)
        anonymized_text = anonymizer.anonymize_cv(extracted_text)
        vector_engine.upsert_cv(candidate_id=cand_id, redacted_text=anonymized_text)

    print("\n--- 3. Testing Semantic Vector Search ---")
    search_query = "Looking for a cloud infrastructure engineer with AWS experience"
    print(f"Search Query: '{search_query}'\n")

    search_results = vector_engine.search_candidates(query=search_query, top_k=2)

    for match in search_results.matches:
        print(f"Match ID: {match.id} | Score: {match.score:.4f}")
        print(f"Sanitized Resume Content:\n{match.metadata['sanitized_text']}")
        print("-" * 50)