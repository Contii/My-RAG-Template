import yaml
import time
from retriever.semantic_retriever import RetrieverStub
from generator.generator import GeneratorStub, LLMGenerator
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
            retrieval_config = self.config.get("retrieval", {})
            from retriever.smart_retriever import SmartRetriever
            self.retriever = SmartRetriever(
                data_path=self.data_path,
                embeddings_path=retrieval_config.get("embeddings_path", "data/embeddings"),
                model_name=retrieval_config.get("model_name", "all-MiniLM-L6-v2"),
                top_k=retrieval_config.get("top_k", 3),
                use_reranking=retrieval_config.get("use_reranking", True),
                use_cache=retrieval_config.get("use_cache", True),
                use_filters=retrieval_config.get("use_filters", True),
                reranker_model=retrieval_config.get("reranker_model", "ms-marco-MiniLM-L-12-v2"),
                rerank_top_k=retrieval_config.get("rerank_top_k", 10),
                cache_ttl_hours=retrieval_config.get("cache_ttl_hours", 24),
                min_score_threshold=retrieval_config.get("min_score_threshold", 0.3)
            )
        elif self.retriever_type == "filtered":
            retrieval_config = self.config.get("retrieval", {})
            from retriever.filtered_retriever import FilteredRetriever
            self.retriever = FilteredRetriever(
                data_path=self.data_path,
                embeddings_path=retrieval_config.get("embeddings_path", "data/embeddings"),
                model_name=retrieval_config.get("model_name", "all-MiniLM-L6-v2"),
                reranker_model=retrieval_config.get("reranker_model", "ms-marco-MiniLM-L-12-v2"),
                top_k=retrieval_config.get("top_k", 3),
                rerank_top_k=retrieval_config.get("rerank_top_k", 10),
                cache_ttl_hours=retrieval_config.get("cache_ttl_hours", 24),
                enable_cache=retrieval_config.get("enable_cache", True),
                min_score_threshold=retrieval_config.get("min_score_threshold", 0.3)
            )
        elif self.retriever_type == "cache":
            retrieval_config = self.config.get("retrieval", {})
            from retriever.cache_retriever import CacheRetriever
            self.retriever = CacheRetriever(
                data_path=self.data_path,
                embeddings_path=retrieval_config.get("embeddings_path", "data/embeddings"),
                model_name=retrieval_config.get("model_name", "all-MiniLM-L6-v2"),
                reranker_model=retrieval_config.get("reranker_model", "ms-marco-MiniLM-L-12-v2"),
                top_k=retrieval_config.get("top_k", 3),
                rerank_top_k=retrieval_config.get("rerank_top_k", 10),
                cache_ttl_hours=retrieval_config.get("cache_ttl_hours", 24),
                enable_cache=retrieval_config.get("enable_cache", True)
            )
        elif self.retriever_type == "reranking":
            retrieval_config = self.config.get("retrieval", {})
            from retriever.reranking_retriever import RerankingRetriever
            self.retriever = RerankingRetriever(
                data_path=self.data_path,
                embeddings_path=retrieval_config.get("embeddings_path", "data/embeddings"),
                model_name=retrieval_config.get("model_name", "all-MiniLM-L6-v2"),
                reranker_model=retrieval_config.get("reranker_model", "ms-marco-MiniLM-L-12-v2"),
                top_k=retrieval_config.get("top_k", 3),
                rerank_top_k=retrieval_config.get("rerank_top_k", 10)
            )
        elif self.retriever_type == "semantic":
            retrieval_config = self.config.get("retrieval", {})
            from retriever.semantic_retriever import SemanticRetriever
            self.retriever = SemanticRetriever(
                data_path=self.data_path,
                embeddings_path=retrieval_config.get("embeddings_path", "data/embeddings"),
                model_name=retrieval_config.get("model_name", "all-MiniLM-L6-v2"),
                top_k=retrieval_config.get("top_k", 3)
            )
        else:
            self.retriever = RetrieverStub()

        # Initialize generator based on configuration
        if self.generator_type == "llm":
            self.generator = LLMGenerator(
                self.llm_model,
                self.max_tokens,
                self.temperature,
                self.max_gpu_memory
            )
        else:
            self.generator = GeneratorStub()

        logger.info(f"Configured retriever: {self.retriever_type}, generator: {self.generator_type}")

    def run(self, question, filters=None):
        """Run pipeline with optional filters."""
        logger.info("Running RAG pipeline")
        if not isinstance(question, str) or not question.strip():
            logger.error(f"Invalid question received: {question}")
            raise ValueError("Question must be a non-empty string.")
        
        if self.use_rag:
            # Apply filters if using filtered retriever
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
                context = self.retriever.retrieve(question)
                
            print(f"Retriever type: {self.retriever_type}")
            print(f"Generator type: {self.generator_type}")
            print(f"Data path: {self.data_path}")
        else:
            context = []
        
        print(f"Using LLM model: {self.llm_model}")
        print(f"Max tokens: {self.max_tokens}")
        print(f"Temperature: {self.temperature}")
        
        answer, generation_time = self.generator.generate(context, question)
        logger.info(f"Answer generated: {answer}")
        log_generation_metrics(logger, generation_time)
        
        # Print metrics and filter info if available
        if hasattr(self.retriever, 'print_metrics'):
            self.retriever.print_metrics()
        
        if hasattr(self.retriever, 'print_filter_info') and filters:
            self.retriever.print_filter_info()
        
        return context, answer