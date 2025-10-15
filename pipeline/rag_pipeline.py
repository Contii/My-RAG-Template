import yaml
import time
from retriever.semantic_retriever import RetrieverStub
from generator.generator import GeneratorStub, HuggingFaceGenerator
from logger.logger import get_logger, log_generation_metrics

logger = get_logger("pipeline")


class RAGPipeline:
    """
    Pipeline for Retrieval-Augmented Generation (RAG).
    Loads configuration from YAML and orchestrates retriever and generator components.
    """

    def __init__(self, use_rag=True, config_path="config/config.yaml"):
        logger.info("Initializing RAGPipeline")
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
            logger.info("Configuration loaded successfully")
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise RuntimeError(f"Failed to load config: {e}")
        
        self.use_rag = use_rag
        self.retriever_type = self.config.get("retriever_type", "stub")
        self.generator_type = self.config.get("generator_type", "stub")
        self.llm_model = self.config.get("llm_model", "default-llm")
        self.max_tokens = self.config.get("max_tokens", 100)
        self.data_path = self.config.get("data_path", "data/")
        self.temperature = self.config.get("temperature", 1.0)
        self.max_gpu_memory = self.config.get("max_gpu_memory", "3.8GB")

        # Initialize retriever based on configuration
        if self.retriever_type == "smart":
            # SmartRetriever handles ALL functionality through configuration
            retrieval_config = self.config.get("retrieval", {})
            from retriever.smart_retriever import SmartRetriever
            self.retriever = SmartRetriever(
                data_path=self.data_path,
                embeddings_path=retrieval_config.get("embeddings_path", "data/embeddings"),
                model_name=retrieval_config.get("model_name", "all-MiniLM-L6-v2"),
                top_k=retrieval_config.get("top_k", 3),
                # All components controlled via config flags
                use_reranking=retrieval_config.get("use_reranking", True),
                use_cache=retrieval_config.get("use_cache", True),
                use_filters=retrieval_config.get("use_filters", True),
                use_metrics=retrieval_config.get("use_metrics", True),
                # Component configurations
                reranker_model=retrieval_config.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L12-v2"),
                rerank_top_k=retrieval_config.get("rerank_top_k", 10),
                cache_ttl_hours=retrieval_config.get("cache_ttl_hours", 24),
                min_score_threshold=retrieval_config.get("min_score_threshold", 0.3)
            )
            logger.info("SmartRetriever initialized with all optional components")

        # Legacy retrievers
        elif self.retriever_type == "filtered":
            retrieval_config = self.config.get("retrieval", {})
            from retriever.filtered_retriever import FilteredRetriever
            self.retriever = FilteredRetriever(
                data_path=self.data_path,
                embeddings_path=retrieval_config.get("embeddings_path", "data/embeddings"),
                model_name=retrieval_config.get("model_name", "all-MiniLM-L6-v2"),
                reranker_model=retrieval_config.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L12-v2"),
                top_k=retrieval_config.get("top_k", 3),
                rerank_top_k=retrieval_config.get("rerank_top_k", 10),
                cache_ttl_hours=retrieval_config.get("cache_ttl_hours", 24),
                enable_cache=retrieval_config.get("enable_cache", True),
                min_score_threshold=retrieval_config.get("min_score_threshold", 0.3)
            )
            logger.info("Legacy FilteredRetriever initialized")
            
        elif self.retriever_type == "cache":
            retrieval_config = self.config.get("retrieval", {})
            from retriever.cache_retriever import CacheRetriever
            self.retriever = CacheRetriever(
                data_path=self.data_path,
                embeddings_path=retrieval_config.get("embeddings_path", "data/embeddings"),
                model_name=retrieval_config.get("model_name", "all-MiniLM-L6-v2"),
                reranker_model=retrieval_config.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L12-v2"),
                top_k=retrieval_config.get("top_k", 3),
                rerank_top_k=retrieval_config.get("rerank_top_k", 10),
                cache_ttl_hours=retrieval_config.get("cache_ttl_hours", 24),
                enable_cache=retrieval_config.get("enable_cache", True)
            )
            logger.info("Legacy CacheRetriever initialized")
            
        elif self.retriever_type == "reranking":
            retrieval_config = self.config.get("retrieval", {})
            from retriever.reranking_retriever import RerankingRetriever
            self.retriever = RerankingRetriever(
                data_path=self.data_path,
                embeddings_path=retrieval_config.get("embeddings_path", "data/embeddings"),
                model_name=retrieval_config.get("model_name", "all-MiniLM-L6-v2"),
                reranker_model=retrieval_config.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L12-v2"),
                top_k=retrieval_config.get("top_k", 3),
                rerank_top_k=retrieval_config.get("rerank_top_k", 10)
            )
            logger.info("Legacy RerankingRetriever initialized")
            
        elif self.retriever_type == "semantic":
            retrieval_config = self.config.get("retrieval", {})
            from retriever.semantic_retriever import SemanticRetriever
            self.retriever = SemanticRetriever(
                data_path=self.data_path,
                embeddings_path=retrieval_config.get("embeddings_path", "data/embeddings"),
                model_name=retrieval_config.get("model_name", "all-MiniLM-L6-v2"),
                top_k=retrieval_config.get("top_k", 3)
            )
            logger.info("Legacy SemanticRetriever initialized")
            
        else:
            self.retriever = RetrieverStub()
            logger.info("Stub retriever initialized")

        # Initialize generator based on configuration
        if self.generator_type == "llm":
            self.generator = HuggingFaceGenerator(
                self.llm_model,
                self.max_tokens,
                self.temperature,
                self.max_gpu_memory
            )
        else:
            self.generator = GeneratorStub()

        logger.info(f"Pipeline configured - Retriever: {self.retriever_type}, Generator: {self.generator_type}")

    def run(self, question, filters=None):
        """Run pipeline with optional filters."""
        logger.info("Running RAG pipeline")
        if not isinstance(question, str) or not question.strip():
            logger.error(f"Invalid question received: {question}")
            raise ValueError("Question must be a non-empty string.")
        
        context = []
        
        if self.use_rag:
            # SmartRetriever and FilteredRetriever support filters
            if hasattr(self.retriever, 'get_available_file_types') and filters:
                context = self.retriever.retrieve(
                    question,
                    file_types=filters.get('file_types'),
                    sources=filters.get('sources'),
                    date_from=filters.get('date_from'),
                    date_to=filters.get('date_to'),
                    min_score=filters.get('min_score')
                )
            else:
                # Basic retrieve for legacy retrievers
                context = self.retriever.retrieve(question)
        
        # Show system info
        print(f"Using LLM model: {self.llm_model}")
        print(f"Max tokens: {self.max_tokens}")
        print(f"Temperature: {self.temperature}")
        
        # Generate answer
        answer, generation_time = self.generator.generate(context, question)
        logger.info(f"Answer generated: {answer}")
        log_generation_metrics(logger, generation_time)
        
        # Print metrics if SmartRetriever with metrics enabled
        if hasattr(self.retriever, 'print_metrics_dashboard'):
            self.retriever.print_metrics_dashboard()
        
        # Performance insights for SmartRetriever
        if hasattr(self.retriever, 'get_performance_insights'):
            insights = self.retriever.get_performance_insights()
            if insights and any('Metrics tracking is disabled' not in insight for insight in insights):
                print("\n💡 PERFORMANCE INSIGHTS:")
                for insight in insights:
                    print(f"   {insight}")
        
        # Print filter info if available (after insights)
        if hasattr(self.retriever, 'print_filter_info') and filters:
            self.retriever.print_filter_info()
        
        return context, answer

    def save_session_metrics(self, filepath="logs/session_report.txt"):
        """Save session metrics if SmartRetriever with metrics is being used."""
        if hasattr(self.retriever, 'save_metrics_report'):
            self.retriever.save_metrics_report(filepath)
            logger.info(f"Session metrics saved to {filepath}")
        else:
            logger.info("Metrics not available for current retriever type")

    def get_retriever_info(self):
        """Get information about current retriever configuration."""
        info = {
            'type': self.retriever_type,
            'data_path': self.data_path
        }
        
        # Get SmartRetriever specific info
        if hasattr(self.retriever, 'use_reranking'):
            info['components'] = {
                'reranking': self.retriever.use_reranking,
                'cache': self.retriever.use_cache,
                'filters': self.retriever.use_filters,
                'metrics': self.retriever.use_metrics if hasattr(self.retriever, 'use_metrics') else False
            }
        
        # Get available options
        if hasattr(self.retriever, 'get_available_file_types'):
            info['available_file_types'] = self.retriever.get_available_file_types()
            info['available_sources'] = self.retriever.get_available_sources()
        
        return info