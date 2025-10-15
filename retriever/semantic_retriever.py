import os
import pickle
import yaml
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.document_parsers import ParserFactory
from retriever.faiss_index import FAISSIndex
from logger.logger import get_logger


logger = get_logger("semantic_retriever")


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
                 model_name="all-MiniLM-L6-v2", top_k=3, config_path="config/config.yaml"):
        self.data_path = data_path
        self.embeddings_path = embeddings_path
        self.model_name = model_name
        self.top_k = top_k
        self.config = self._load_config(config_path)

        logger.info(f"Initializing SemanticRetriever with model: {model_name}")
        logger.info(f"Supported formats: {ParserFactory.supported_extensions()}")
        print(f"Initializing SemanticRetriever with model: {model_name}")
        
        # Load sentence transformer model
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        faiss_config = self.config.get('retrieval', {}).get('faiss', None)
        self.faiss_index = FAISSIndex(dimension=self.embedding_dim, config=faiss_config)

        logger.info(f"FAISS configuration: {self.faiss_index.get_stats()}")

        # Initialize document storage
        self.documents = []
        self.embeddings = None
        
        # Create embeddings directory if it doesn't exist
        os.makedirs(self.embeddings_path, exist_ok=True)
        
        # Load or create embeddings
        self._load_or_create_embeddings()
    
    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return {}
        
    def _load_documents(self):
        """Load documents from multiple formats using appropriate parsers."""
        logger.info(f"Loading documents from {self.data_path}")
        
        if not os.path.exists(self.data_path):
            logger.warning(f"Data path {self.data_path} does not exist")
            return
        
        documents = []
        supported_extensions = ParserFactory.supported_extensions()

        for filename in os.listdir(self.data_path):
            file_path = os.path.join(self.data_path, filename)
            
            # Skip directories
            if os.path.isdir(file_path):
                continue
            
            # Check if file extension is supported
            _, ext = os.path.splitext(filename.lower())
            if ext not in supported_extensions:
                logger.warning(f"Skipping unsupported file: {filename}")
                continue
            
            # Parse document
            parser = ParserFactory.get_parser(file_path)
            if parser:
                content = parser.parse(file_path)
                
                if content:
                    # Split into chunks
                    chunks = self._chunk_text(content)
                    for i, chunk in enumerate(chunks):
                        doc_info = {
                            'content': chunk,
                            'source': filename,
                            'chunk_id': i,
                            'file_type': ext,
                            'file_path': file_path
                        }
                        documents.append(doc_info)
                    logger.info(f"Loaded {len(chunks)} chunks from {filename} ({ext})")
                else:
                    logger.warning(f"No content extracted from {filename}")
        
        self.documents = documents
        logger.info(f"Total documents loaded: {len(self.documents)} from {len(set(d['source'] for d in documents))} files")
    
    
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
        faiss_file = os.path.join(self.embeddings_path, "faiss_index.pkl")
        
        # Check if embeddings exist
        if (os.path.exists(embeddings_file) and 
            os.path.exists(documents_file) and 
            os.path.exists(faiss_file)):
            logger.info("Loading existing embeddings...")
            print("Loading existing embeddings...")
            try:
                with open(documents_file, 'rb') as f:
                    self.documents = pickle.load(f)
                
                with open(embeddings_file, 'rb') as f:
                    self.embeddings = pickle.load(f)
                
                # Load FAISS index using FAISSIndex class
                self.faiss_index.load(faiss_file)
                
                stats = self.faiss_index.get_stats()
                logger.info(f"FAISS index loaded: {stats}")

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
        
        # Add embeddings to FAISS index (handles normalization internally)
        embeddings_array = np.array(self.embeddings).astype('float32')
        self.faiss_index.add_embeddings(embeddings_array)
        
        stats = self.faiss_index.get_stats()
        logger.info(f"FAISS index built: {stats}")
        
        # Save embeddings and index
        try:
            with open(documents_file, 'wb') as f:
                pickle.dump(self.documents, f)
            
            with open(embeddings_file, 'wb') as f:
                pickle.dump(self.embeddings, f)
            
            # Save FAISS index using FAISSIndex class
            self.faiss_index.save(faiss_file)
            
            logger.info("Embeddings and index saved successfully")
        except Exception as e:
            logger.error(f"Error saving embeddings: {e}")
    
    def retrieve(self, query):
        """Retrieve most relevant documents for the given query."""
        logger.info(f"Retrieving documents for query: {query[:50]}...")
        print(f"\nRetrieving documents for query: {query[:50]}...")
        
        stats = self.faiss_index.get_stats()
        logger.debug(f"FAISS stats - vectors: {stats['num_vectors']}, index_type: {stats['index_type']}")
        
        if not self.documents or self.faiss_index.is_empty():
            logger.warning("No documents or index available")
            return ["No documents available"]
        
        try:
            # Generate query embedding
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            query_embedding = query_embedding.astype('float32')
            
            # Search using FAISSIndex
            indices, scores = self.faiss_index.search(query_embedding, self.top_k)
            
            logger.debug(f"Search completed - top_k: {self.top_k}, results: {len(indices)}")

            # Prepare results
            results = []
            for i, (idx, score) in enumerate(zip(indices, scores)):
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

    def get_index_stats(self):
        """Get FAISS index statistics."""
        return self.faiss_index.get_stats()
    
    def rebuild_index(self):
        """Rebuild the index from scratch."""
        logger.info("Rebuilding index...")
        self.faiss_index.clear()
        self.embeddings = None
        self._load_or_create_embeddings()
        logger.info("Index rebuilt successfully")