import yaml
import logging
from retriever.retriever import RetrieverStub
from generator.generator import GeneratorStub, LLMGenerator


class RAGPipeline:
    """
    Pipeline for Retrieval-Augmented Generation (RAG).
    Loads configuration from YAML and orchestrates retriever and generator components.
    """

    def __init__(self, use_rag=True, config_path="config/config.yaml"):
        logging.info("Initializing RAGPipeline")
        try:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)
            logging.info("Configuration loaded successfully")
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            raise RuntimeError(f"Failed to load config: {e}")
        self.use_rag = use_rag
        self.retriever_type = self.config.get("retriever_type", "stub")
        self.generator_type = self.config.get("generator_type", "stub")
        self.llm_model = self.config.get("llm_model", "default-llm")
        self.max_tokens = self.config.get("max_tokens", 100)
        self.data_path = self.config.get("data_path", "data/")
        self.temperature = self.config.get("temperature", 1.0)
        self.retriever = RetrieverStub()
        if self.generator_type == "llm":
            self.generator = LLMGenerator(self.llm_model, self.max_tokens, self.temperature)
        else:
            self.generator = GeneratorStub()
        logging.info(f"Configured retriever: {self.retriever_type}, generator: {self.generator_type}")

    def run(self, question):
        logging.info("Running RAG pipeline")
        if not isinstance(question, str) or not question.strip():
            logging.error(f"Invalid question received: {question}")
            raise ValueError("Question must be a non-empty string.")
        if self.use_rag:
            context = self.retriever.retrieve(question)
            answer = self.generator.generate(context, question)
            logging.info(f"Retriever type: {self.retriever_type}\n Generator type: {self.generator_type}\n Data path: {self.data_path}")
            print(f"Retriever type: {self.retriever_type}\n Generator type: {self.generator_type}\n Data path: {self.data_path}")
        else:
            context = []
        logging.info(f"Using LLM model: {self.llm_model}\n Max tokens: {self.max_tokens}\n Temperature: {self.temperature}")
        print(f"Using LLM model: {self.llm_model}\n Max tokens: {self.max_tokens}\n Temperature: {self.temperature}")
        answer = self.generator.generate(context, question)
        return context, answer
