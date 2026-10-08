import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

load_dotenv()


class VectorSearchEngine:
    def __init__(self, index_name="cp001-anonymized-cvs"):
        print("Loading free local embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY environment variable is missing in .env file.")

        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name

        existing_indexes = [index.name for index in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            print(f"Creating Pinecone index: {self.index_name}...")
            self.pc.create_index(
                name=self.index_name,
                dimension=384,  # Dimension for all-MiniLM-L6-v2
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
        self.index = self.pc.Index(self.index_name)

    def generate_embedding(self, text: str) -> list[float]:
        """Generates dense vector embeddings locally without API costs."""
        return self.embedding_model.encode(text).tolist()

    def upsert_cv(self, candidate_id: str, redacted_text: str):
        """Generates vector embedding and stores the anonymized text in Pinecone."""
        if not redacted_text.strip():
            return
        
        # Clean ID format
        clean_id = candidate_id.replace(" ", "_").encode('ascii', 'ignore').decode('ascii')
        vector = self.generate_embedding(redacted_text)
        
        self.index.upsert(
            vectors=[
                {
                    "id": clean_id,
                    "values": vector,
                    "metadata": {"sanitized_text": redacted_text}
                }
            ]
        )
        print(f"--> Candidate '{clean_id}' successfully indexed in Pinecone.")

    def search_candidates(self, query: str, top_k: int = 3):
        """Performs vector search across indexed anonymized resumes."""
        query_vector = self.generate_embedding(query)
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )
        return results