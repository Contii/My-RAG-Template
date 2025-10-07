import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from logger.logger import get_logger

logger = get_logger("retriever")


class RetrieverStub:
    """
    Stub retriever that returns a static example text.
    """

    def retrieve(self, query):
        logger.info(f"Retrieving context for query: {query}")
        return ["Sample text about RAG"]

class SemanticRetriever:
    """
    Semantic retriever using sentence-transformers and FAISS for similarity search.
    """
    
    def __init__(self, data_path="data/documents", embeddings_path="data/embeddings", 
                 model_name="all-MiniLM-L6-v2", top_k=3):
        self.data_path = data_path
        self.embeddings_path = embeddings_path
        self.model_name = model_name
        self.top_k = top_k
        
        logger.info(f"Initializing SemanticRetriever with model: {model_name}")
        print(f"Initializing SemanticRetriever with model: {model_name}")
        
        # Load sentence transformer model
        self.model = SentenceTransformer(model_name)
        
        # Initialize document storage
        self.documents = []
        self.embeddings = None
        self.index = None
        
        # Create embeddings directory if it doesn't exist
        os.makedirs(self.embeddings_path, exist_ok=True)
        
        # Load or create embeddings
        self._load_or_create_embeddings()
    
    def _load_documents(self):
        """Load documents from the data directory."""
        logger.info(f"Loading documents from {self.data_path}")
        
        if not os.path.exists(self.data_path):
            logger.warning(f"Data path {self.data_path} does not exist")
            return
        
        documents = []
        for filename in os.listdir(self.data_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(self.data_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            # Simple sentence splitting into chunks
                            chunks = self._chunk_text(content)
                            for i, chunk in enumerate(chunks):
                                doc_info = {
                                    'content': chunk,
                                    'source': filename,
                                    'chunk_id': i
                                }
                                documents.append(doc_info)
                    logger.info(f"Loaded {len(chunks)} chunks from {filename}")
                except Exception as e:
                    logger.error(f"Error loading file {filename}: {e}")
        
        self.documents = documents
        logger.info(f"Total documents loaded: {len(self.documents)}")
    
    def _chunk_text(self, text, chunk_size=300, overlap=50):
        """Simple text chunking by character count."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + chunk_size // 2:
                    chunk = text[start:start + break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= len(text):
                break
        
        return [chunk for chunk in chunks if len(chunk.strip()) > 20]
    
    def _load_or_create_embeddings(self):
        """Load existing embeddings or create new ones."""
        embeddings_file = os.path.join(self.embeddings_path, "embeddings.pkl")
        documents_file = os.path.join(self.embeddings_path, "documents.pkl")
        index_file = os.path.join(self.embeddings_path, "faiss_index.bin")
        
        # Check if embeddings exist
        if (os.path.exists(embeddings_file) and 
            os.path.exists(documents_file) and 
            os.path.exists(index_file)):
            logger.info("Loading existing embeddings...")
            print("Loading existing embeddings...")
            try:
                with open(documents_file, 'rb') as f:
                    self.documents = pickle.load(f)
                
                with open(embeddings_file, 'rb') as f:
                    self.embeddings = pickle.load(f)
                
                self.index = faiss.read_index(index_file)
                logger.info(f"Loaded {len(self.documents)} documents with embeddings")
                print(f"Loaded {len(self.documents)} documents with embeddings successfully.")
                return
            except Exception as e:
                logger.error(f"Error loading embeddings: {e}")
        
        # Create new embeddings
        logger.info("Creating new embeddings...")
        self._load_documents()
        
        if not self.documents:
            logger.warning("No documents found. Using fallback context.")
            self.documents = [{"content": "Sample text about RAG", "source": "fallback", "chunk_id": 0}]
        
        # Generate embeddings
        texts = [doc['content'] for doc in self.documents]
        logger.info(f"Generating embeddings for {len(texts)} text chunks...")
        
        self.embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        # Save embeddings and index
        try:
            with open(documents_file, 'wb') as f:
                pickle.dump(self.documents, f)
            
            with open(embeddings_file, 'wb') as f:
                pickle.dump(self.embeddings, f)
            
            faiss.write_index(self.index, index_file)
            logger.info("Embeddings and index saved successfully")
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
    
    def retrieve(self, query):
        """Retrieve most relevant documents for the given query."""
        logger.info(f"Retrieving documents for query: {query[:50]}...")
        print(f"Retrieving documents for query: {query[:50]}...")
        
        if not self.documents or self.index is None:
            logger.warning("No documents or index available")
            return ["No documents available"]
        
        try:
            # Generate query embedding
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Search for similar documents
            scores, indices = self.index.search(query_embedding, self.top_k)
            
            # Prepare results
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    result = f"[Score: {score:.3f}] {doc['content']}"
                    results.append(result)
                    logger.info(f"Retrieved doc {i+1}: score={score:.3f}, source={doc['source']}")
                    print(f"Retrieved doc {i+1}: score={score:.3f}, source={doc['source']}")
            
            logger.info(f"Retrieved {len(results)} relevant documents")
            print(f"Retrieved {len(results)} relevant documents")
            return results if results else ["No relevant documents found"]
            
        except Exception as e:
            logger.error(f"Error during retrieval: {e}")
            return [f"Error during retrieval: {e}"]