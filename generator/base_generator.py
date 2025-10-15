from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional


"""
Base abstraction for LLM generators.
Defines the interface that all generator implementations must follow.
"""


class BaseGenerator(ABC):
    """
    Abstract base class for all LLM generators.
    
    All generator implementations must inherit from this class and implement
    the required abstract methods.
    """
    
    @abstractmethod
    def generate(self, context: list, question: str) -> Tuple[str, float]:
        """
        Generate an answer given context and question.
        
        Args:
            context: List of context strings (can be empty for LLM-only mode)
            question: User question string
            
        Returns:
            Tuple of (answer: str, generation_time: float)
            
        Raises:
            Exception: If generation fails
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Return metadata about the model/generator.
        
        Returns:
            Dictionary with model information:
                - type: str (e.g., 'huggingface', 'ollama', 'stub')
                - model_id/name: str
                - Additional implementation-specific info
        """
        pass
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text (optional override).
        
        Default implementation uses simple whitespace splitting.
        Subclasses with proper tokenizers should override this.
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Estimated token count
        """
        return len(text.split())
    
    def get_metrics(self) -> Optional[Any]:
        """
        Get generator metrics if available.
        
        Returns:
            GeneratorMetrics instance or None if not implemented
        """
        return getattr(self, 'metrics', None)
    
    def __repr__(self) -> str:
        """String representation of the generator."""
        info = self.get_model_info()
        return f"{self.__class__.__name__}(type={info.get('type')}, model={info.get('model_id', info.get('name', 'unknown'))})"