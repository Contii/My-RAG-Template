import faiss
from sentence_transformers import CrossEncoder
from retriever.retriever import SemanticRetriever
from logger.logger import get_logger

logger = get_logger("reranking_retriever")

class RerankingRetriever(SemanticRetriever):
    """
    Semantic retriever with basic reranking capability.
    """
    
    def __init__(self, data_path="data/documents", embeddings_path="data/embeddings", 
                 model_name="all-MiniLM-L6-v2", reranker_model="ms-marco-MiniLM-L-12-v2", 
                 top_k=3, rerank_top_k=10):
        super().__init__(data_path, embeddings_path, model_name, top_k)
        
        # Initialize reranker
        logger.info(f"Loading reranker model: {reranker_model}")
        self.reranker = CrossEncoder(reranker_model)
        self.rerank_top_k = rerank_top_k
        logger.info(f"Reranker initialized successfully")
    
    def retrieve(self, query):
        """Retrieve with reranking."""
        logger.info(f"Starting reranking retrieval for query: {query[:50]}...")
        
        if not self.documents or self.index is None:
            logger.warning("No documents or index available")
            return ["No documents available"]
        
        try:
            # Get more candidates for reranking (3x the final amount)
            candidates_count = min(self.rerank_top_k, len(self.documents))
            
            # Generate query embedding
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Search for similar documents
            scores, indices = self.index.search(query_embedding, candidates_count)
            
            # Collect candidates
            candidates = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    candidates.append({
                        'content': doc['content'],
                        'source': doc['source'],
                        'initial_score': score,
                        'index': idx
                    })
            
            # Rerank candidates
            logger.info(f"Reranking {len(candidates)} candidates")
            reranked = self._rerank_candidates(query, candidates)
            
            # Format final results
            results = []
            for i, candidate in enumerate(reranked[:self.top_k]):
                result = f"[Score: {candidate['rerank_score']:.3f}] {candidate['content']}"
                results.append(result)
                logger.info(f"Final doc {i+1}: rerank_score={candidate['rerank_score']:.3f}, source={candidate['source']}")
            
            logger.info(f"Retrieved {len(results)} reranked documents")
            return results
            
        except Exception as e:
            logger.error(f"Error during reranking retrieval: {e}")
            return [f"Error during retrieval: {e}"]
    
    def _rerank_candidates(self, query, candidates):
        """Rerank candidates using cross-encoder."""
        if len(candidates) <= 1:
            return candidates
        
        # Prepare pairs for reranking
        query_doc_pairs = [(query, candidate['content']) for candidate in candidates]
        
        # Get reranking scores
        rerank_scores = self.reranker.predict(query_doc_pairs)
        
        # Add reranking scores to candidates
        for candidate, score in zip(candidates, rerank_scores):
            candidate['rerank_score'] = float(score)
        
        # Sort by reranking score (descending)
        reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
        
        logger.info(f"Reranking complete. Top score: {reranked[0]['rerank_score']:.3f}")
        return reranked