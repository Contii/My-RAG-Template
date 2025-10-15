import yaml
import time
import torch
from retriever.semantic_retriever import RetrieverStub
from generator.generator import GeneratorStub
from generator.generator_factory import GeneratorFactory
from logger.logger import get_logger, log_generation_metrics

logger = get_logger("pipeline")


class RAGPipeline:
    """
    Pipeline for Retrieval-Augmented Generation (RAG).
    Loads configuration from YAML and orchestrates retriever and generator components.
    """

    def __init__(self, use_rag=True, config_path="config/config.yaml", generator=None):
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
        self.data_path = self.config.get("data_path", "data/")
        self.config_path = config_path

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

        if generator:
            # Use provided generator instance
            self.generator = generator
            self.current_model = 'provided'
            self.available_models = []
            self._model_configs = {}
            logger.info(f"Using provided generator: {self.generator.get_model_info()['type']}")
        else:
            # Create generator from config using factory
            generator_config = self.config.get('generator', {})
            
            # Check for multi-model config
            models = generator_config.get('models', {})
            
            if models:
                # Multi-model configuration
                self.available_models = list(models.keys())
                self.current_model = generator_config.get('active_model', self.available_models[0])
                self._model_configs = models
                
                # Load active model
                model_config = models[self.current_model]
                self.generator = GeneratorFactory.create_generator(model_config)
                
                logger.info(f"Multi-model setup: {len(self.available_models)} models available")
                logger.info(f"Active model: {self.current_model}")
                logger.info(f"Available models: {', '.join(self.available_models)}")
            
            else:
                # Single model configuration (legacy or simple)
                self.available_models = []
                self.current_model = 'default'
                self._model_configs = {}
                
                # Fallback to legacy config format if new format not found
                if not generator_config or 'type' not in generator_config:
                    logger.warning("New generator config not found, attempting legacy format")
                    legacy_generator_type = self.config.get("generator_type", "stub")
                    
                    if legacy_generator_type == "llm":
                        generator_config = {
                            'type': 'huggingface',
                            'model_id': self.config.get("llm_model", "microsoft/bitnet-b1.58-2B-4T"),
                            'max_tokens': self.config.get("max_tokens", 250),
                            'temperature': self.config.get("temperature", 0.7),
                            'max_gpu_memory': self.config.get("max_gpu_memory", "3.8GB")
                        }
                    else:
                        generator_config = {'type': 'stub'}
                
                self.generator = GeneratorFactory.create_generator(generator_config)
                logger.info(f"Single model configuration loaded")

        # Store generator info for display
        self.generator_info = self.generator.get_model_info()

        logger.info(f"Pipeline configured - Retriever: {self.retriever_type}, Generator: {self.generator_type}")

    def switch_model(self, model_name: str):
        """
        Switch to a different model without restarting the pipeline.
        
        Args:
            model_name: Name of the model to switch to
            
        Raises:
            ValueError: If model not available or multi-model not configured
        """
        if not self.available_models:
            raise ValueError(
                "Model switching not available. "
                "Configure multiple models in config.yaml under generator.models"
            )
        
        if model_name not in self.available_models:
            raise ValueError(
                f"Model '{model_name}' not available. "
                f"Available models: {', '.join(self.available_models)}"
            )
        
        if model_name == self.current_model:
            logger.info(f"Model '{model_name}' is already active")
            return
        
        logger.info(f"Switching model from '{self.current_model}' to '{model_name}'")
        print(f"\n🔄 Switching model from '{self.current_model}' to '{model_name}'...")
        
        try:
            # Unload current model
            del self.generator
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Load new model
            model_config = self._model_configs[model_name]
            self.generator = GeneratorFactory.create_generator(model_config)
            self.current_model = model_name
            self.generator_info = self.generator.get_model_info()
            
            logger.info(f"Successfully switched to model: {model_name}")
            print(f"✅ Model switched successfully to '{model_name}'")
            print(f"   Model ID: {self.generator_info.get('model_id', 'N/A')}")
            print(f"   Quantization: {self.generator_info.get('quantization', 'none')}")
            
        except Exception as e:
            logger.error(f"Error switching model: {e}")
            print(f"❌ Error switching model: {e}")
            # Try to reload previous model
            try:
                model_config = self._model_configs[self.current_model]
                self.generator = GeneratorFactory.create_generator(model_config)
                logger.info(f"Reverted to previous model: {self.current_model}")
            except:
                raise RuntimeError(f"Failed to switch model and couldn't revert: {e}")

    def list_models(self) -> dict:
        """
        List available models and current active model.
        
        Returns:
            Dictionary with 'current' and 'available' keys
        """
        return {
            'current': self.current_model,
            'available': self.available_models,
            'multi_model_enabled': len(self.available_models) > 0
        }
          
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
        
        # Show generator info from stored metadata
        print(f"\n🤖 Using {self.generator_info['type']} generator: {self.generator_info.get('model_id', self.generator_info.get('name', 'unknown'))}")
        if self.available_models:
            print(f"   Active model profile: {self.current_model}")
        print(f"   Max tokens: {self.generator_info.get('max_tokens', 'N/A')}")
        print(f"   Temperature: {self.generator_info.get('temperature', 'N/A')}")
        if 'quantization' in self.generator_info and self.generator_info['quantization']:
            print(f"   Quantization: {self.generator_info['quantization']}")
        
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
        
        return 
    
    def get_generator_info(self):
        """Get information about current generator configuration."""
        info = self.generator_info.copy()
        if self.available_models:
            info['current_model'] = self.current_model
            info['available_models'] = self.available_models
            info['multi_model_enabled'] = True
        else:
            info['multi_model_enabled'] = False
        
        return info