import yaml
from generator.base_generator import BaseGenerator
from generator.generator import GeneratorStub, HuggingFaceGenerator
from logger.logger import get_logger

logger = get_logger("generator_factory")

"""
Factory for creating generator instances from configuration.
Supports multiple generator types: stub, huggingface (future: ollama, gguf, vllm).
"""

class GeneratorFactory:
    """
    Factory for creating generator instances based on configuration.
    
    Supports:
        - stub: GeneratorStub for testing
        - huggingface: HuggingFaceGenerator for local HF models
        - (future) ollama: OllamaGenerator
        - (future) gguf: GGUFGenerator
        - (future) vllm: vLLMGenerator
    """
    
    @staticmethod
    def create_generator(config: dict) -> BaseGenerator:
        """
        Create a generator from configuration dictionary.
        
        Args:
            config: Generator configuration dictionary with at least 'type' key
            
        Returns:
            BaseGenerator instance
            
        Raises:
            ValueError: If generator type is unknown or config is invalid
            
        Example config:
            {
                'type': 'huggingface',
                'model_id': 'microsoft/bitnet-b1.58-2B-4T',
                'max_tokens': 250,
                'temperature': 0.7,
                'max_gpu_memory': '3.8GB'
            }
        """
        if not config:
            raise ValueError("Generator config cannot be empty")
        
        generator_type = config.get('type', '').lower()
        
        if not generator_type:
            raise ValueError("Generator config must specify 'type'")
        
        logger.info(f"Creating generator of type: {generator_type}")
        
        # Stub generator
        if generator_type == 'stub':
            logger.info("Creating GeneratorStub")
            return GeneratorStub()
        
        # HuggingFace generator
        elif generator_type == 'huggingface':
            model_id = config.get('model_id')
            if not model_id:
                raise ValueError("HuggingFace generator requires 'model_id' in config")
            
            logger.info(f"Creating HuggingFaceGenerator: {model_id}")
            return HuggingFaceGenerator(config=config)
        
        # Future generators (commented examples)
        # elif generator_type == 'ollama':
        #     from generator.ollama_generator import OllamaGenerator
        #     model = config.get('model', 'llama3')
        #     base_url = config.get('base_url', 'http://localhost:11434')
        #     logger.info(f"Creating OllamaGenerator: {model}")
        #     return OllamaGenerator(model=model, base_url=base_url)
        
        # elif generator_type == 'gguf':
        #     from generator.gguf_generator import GGUFGenerator
        #     model_path = config.get('model_path')
        #     if not model_path:
        #         raise ValueError("GGUF generator requires 'model_path' in config")
        #     logger.info(f"Creating GGUFGenerator: {model_path}")
        #     return GGUFGenerator(model_path=model_path, **config)
        
        # Unknown type
        else:
            supported = GeneratorFactory.list_supported_types()
            raise ValueError(
                f"Unknown generator type: '{generator_type}'. "
                f"Supported types: {', '.join(supported)}"
            )
    
    @staticmethod
    def list_supported_types() -> list:
        """
        List all supported generator types.
        
        Returns:
            List of supported generator type strings
        """
        return ['stub', 'huggingface']  # Future: 'ollama', 'gguf', 'vllm'

    @staticmethod
    def list_available_models(config_path="config/config.yaml") -> list:
        """
        List all configured models from config file.
        
        Args:
            config_path: Path to configuration YAML
            
        Returns:
            List of model names
        """
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            gen_config = config.get('generator', {})
            models = gen_config.get('models', {})
            
            if not models:
                # Single model config (legacy)
                if 'model_id' in gen_config:
                    return ['default']
                return []
            
            return list(models.keys())
        
        except Exception as e:
            logger.error(f"Error listing models from {config_path}: {e}")
            return []

    @staticmethod
    def get_model_config(model_name: str, config_path="config/config.yaml") -> dict:
        """
        Get configuration for a specific model.
        
        Args:
            model_name: Name of the model profile
            config_path: Path to configuration YAML
            
        Returns:
            Model configuration dictionary
            
        Raises:
            ValueError: If model not found
        """
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            gen_config = config.get('generator', {})
            models = gen_config.get('models', {})
            
            if not models:
                # Single model config (legacy)
                if model_name == 'default' and 'model_id' in gen_config:
                    return gen_config
                raise ValueError(f"No models configured in {config_path}")
            
            if model_name not in models:
                available = list(models.keys())
                raise ValueError(
                    f"Model '{model_name}' not found. "
                    f"Available models: {', '.join(available)}"
                )
            
            return models[model_name]
        
        except Exception as e:
            logger.error(f"Error getting model config for '{model_name}': {e}")
            raise
    
    @staticmethod
    def get_active_model_name(config_path="config/config.yaml") -> str:
        """
        Get the name of the currently active model.
        
        Args:
            config_path: Path to configuration YAML
            
        Returns:
            Active model name
        """
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            
            gen_config = config.get('generator', {})
            
            # Check for multi-model config
            if 'models' in gen_config:
                return gen_config.get('active_model', list(gen_config['models'].keys())[0])
            
            # Single model (legacy)
            return 'default'
        
        except Exception as e:
            logger.error(f"Error getting active model from {config_path}: {e}")
            return 'default'

    @staticmethod
    def create_from_legacy_config(
        generator_type: str,
        llm_model: str = None,
        max_tokens: int = 250,
        temperature: float = 0.7,
        max_gpu_memory: str = "3.8GB"
    ) -> BaseGenerator:
        """
        Create generator from legacy flat config format (backward compatibility).
        
        Args:
            generator_type: 'stub' or 'llm'
            llm_model: Model ID for HuggingFace
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            max_gpu_memory: Max GPU memory allocation
            
        Returns:
            BaseGenerator instance
        """
        logger.info(f"Creating generator from legacy config: {generator_type}")
        
        if generator_type == 'stub':
            return GeneratorStub()
        
        elif generator_type == 'llm':
            if not llm_model:
                raise ValueError("LLM generator requires llm_model parameter")
            
            config = {
                'type': 'huggingface',
                'model_id': llm_model,
                'max_tokens': max_tokens,
                'temperature': temperature,
                'max_gpu_memory': max_gpu_memory
            }
            return HuggingFaceGenerator(config=config)
        
        else:
            raise ValueError(
                f"Unknown legacy generator type: '{generator_type}'. "
                f"Use 'stub' or 'llm'"
            )