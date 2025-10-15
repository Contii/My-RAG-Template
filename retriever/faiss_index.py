import os
import pickle
import numpy as np
import faiss
from logger.logger import get_logger

logger = get_logger("faiss_index")

"""
FAISS index manager with pickle persistence.
Handles FAISS index operations independently from retrieval logic.
"""

class FAISSIndex:
    """Manages FAISS index operations with persistence."""
    
    def __init__(self, dimension=384, metric="inner_product", config=None):
        """
        Initialize FAISS index.
        
        Args:
            dimension: Embedding dimension
            metric: Distance metric ('inner_product' or 'L2')
            config: Optional dict with FAISS configuration
        """

        if config:
            self.dimension = config.get('dimension', dimension)
            self.metric = config.get('metric', metric)
        else:
            self.dimension = dimension
            self.metric = metric

        self.index = self._create_index()
        self.embeddings_list = []  # Store embeddings for persistence
        
        logger.info(f"FAISSIndex initialized - dimension: {dimension}, metric: {metric}")
    
    def _create_index(self):
        """Create FAISS index based on metric."""
        if self.metric == "inner_product":
            index = faiss.IndexFlatIP(self.dimension)
            logger.debug("Created IndexFlatIP (inner product)")
        elif self.metric == "L2":
            index = faiss.IndexFlatL2(self.dimension)
            logger.debug("Created IndexFlatL2 (L2 distance)")
        else:
            raise ValueError(f"Unknown metric: {self.metric}. Use 'inner_product' or 'L2'")
        
        return index
    
    def add_embeddings(self, embeddings):
        """
        Add embeddings to the index.
        
        Args:
            embeddings: numpy array of shape (n_samples, dimension)
        """
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings, dtype='float32')
        
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype('float32')
        
        # Normalize for cosine similarity if using inner product
        if self.metric == "inner_product":
            faiss.normalize_L2(embeddings)
        
        self.index.add(embeddings)
        self.embeddings_list.append(embeddings)
        
        logger.info(f"Added {embeddings.shape[0]} embeddings to index (total: {self.index.ntotal})")
    
    def search(self, query_embedding, top_k=5):
        """
        Search for nearest neighbors.
        
        Args:
            query_embedding: numpy array of shape (1, dimension) or (dimension,)
            top_k: number of results to return
            
        Returns:
            Tuple of (indices, distances) arrays
        """
        if not isinstance(query_embedding, np.ndarray):
            query_embedding = np.array(query_embedding, dtype='float32')
        
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype('float32')
        
        # Ensure 2D shape (1, dimension)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize for cosine similarity if using inner product
        if self.metric == "inner_product":
            faiss.normalize_L2(query_embedding)
        
        distances, indices = self.index.search(query_embedding, top_k)
        
        logger.debug(f"Search completed - top_k: {top_k}, results: {len(indices[0])}")
        
        return indices[0], distances[0]
    
    def save(self, filepath):
        """
        Save index and embeddings to disk using pickle.
        
        Args:
            filepath: Path to save the index (will create .pkl file)
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Serialize FAISS index
            index_binary = faiss.serialize_index(self.index)
            
            data = {
                'index': index_binary,
                'embeddings_list': self.embeddings_list,
                'dimension': self.dimension,
                'metric': self.metric
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"Index saved to {filepath} ({self.index.ntotal} vectors)")
            
        except Exception as e:
            logger.error(f"Failed to save index to {filepath}: {e}")
            raise
    
    def load(self, filepath):
        """
        Load index and embeddings from disk.
        
        Args:
            filepath: Path to load the index from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Index file not found: {filepath}")
                return False
            
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # Deserialize FAISS index
            self.index = faiss.deserialize_index(data['index'])
            self.embeddings_list = data['embeddings_list']
            self.dimension = data['dimension']
            self.metric = data['metric']
            
            logger.info(f"Index loaded from {filepath} ({self.index.ntotal} vectors)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load index from {filepath}: {e}")
            return False
    
    def get_stats(self):
        """
        Get statistics about the index.
        
        Returns:
            Dictionary with index statistics
        """
        return {
            'num_vectors': self.index.ntotal,
            'dimension': self.dimension,
            'index_type': 'IndexFlatIP' if self.metric == "inner_product" else 'IndexFlatL2',
            'metric': self.metric
        }
    
    def clear(self):
        """Clear the index and embeddings."""
        self.index.reset()
        self.embeddings_list.clear()
        logger.info("Index cleared")
    
    def is_empty(self):
        """Check if index is empty."""
        return self.index.ntotal == 0
    
    def __len__(self):
        """Return number of vectors in index."""
        return self.index.ntotal
    
    def __repr__(self):
        """String representation."""
        return (f"FAISSIndex(vectors={self.index.ntotal}, "
                f"dimension={self.dimension}, metric={self.metric})")