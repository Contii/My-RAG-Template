import os
import json
from pathlib import Path
from bs4 import BeautifulSoup
from docling.document_converter import DocumentConverter
from logger.logger import get_logger

logger = get_logger("parsers")

class DocumentParser:
    """Base class for document parsers."""
    
    def parse(self, file_path):
        raise NotImplementedError

class TxtParser(DocumentParser):
    """Parser for .txt files."""
    
    def parse(self, file_path):
        logger.info(f"Parsing TXT file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            if content:
                logger.info(f"Successfully extracted {len(content)} characters from {file_path}")
                return content
            else:
                logger.warning(f"No content found in {file_path}")
                return ""
            
        except Exception as e:
            logger.error(f"Error parsing TXT file {file_path}: {e}")
            return ""

class DoclingParser(DocumentParser):
    """Universal parser using Docling for PDF, DOCX, PPTX, etc."""
    
    def __init__(self):
        self.converter = DocumentConverter()
    
    def parse(self, file_path):
        logger.info(f"Parsing with Docling: {file_path}")
        try:
            # Docling handles: PDF, DOCX, PPTX, HTML, MD
            result = self.converter.convert(file_path)
            content = result.document.export_to_text()

            if content:
                cleaned_content = content.strip()
                logger.info(f"Successfully extracted {len(cleaned_content)} characters from {file_path}")
                return cleaned_content
            else:
                logger.warning(f"No content extracted from {file_path}")
                return "" # Avoid polluting embeddings.
                
        except Exception as e:
            logger.error(f"Error parsing with Docling {file_path}: {e}")
            return "" # Ignores system breaks.

class HTMLParser(DocumentParser):
    """Parser for HTML files using BeautifulSoup."""
    
    def parse(self, file_path):
        logger.info(f"Parsing HTML file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text and clean it
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            if text:
                logger.info(f"Successfully extracted {len(text)} characters from {file_path}")
                return text
            else:
                logger.warning(f"No text content found in {file_path}")
                return ""
            
        except Exception as e:
            logger.error(f"Error parsing HTML file {file_path}: {e}")
            return ""

class JSONParser(DocumentParser):
    """Parser for .json files."""
    
    def parse(self, file_path):
        logger.info(f"Parsing JSON file: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract text content from JSON
            if isinstance(data, dict):
                content = self._extract_text_from_dict(data)
            elif isinstance(data, list):
                content = self._extract_text_from_list(data)
            else:
                content = str(data)
            
            if content:
                cleaned_content = content.strip()
                logger.info(f"Successfully extracted {len(cleaned_content)} characters from {file_path}")
                return cleaned_content
            else:
                logger.warning(f"No content extracted from {file_path}")
                return ""
            
        except Exception as e:
            logger.error(f"Error parsing JSON file {file_path}: {e}")
            return ""
    
    def _extract_text_from_dict(self, data):
        """Extract text from dictionary recursively."""
        text_parts = []
        for key, value in data.items():
            if isinstance(value, str):
                text_parts.append(f"{key}: {value}")
            elif isinstance(value, (int, float)):
                text_parts.append(f"{key}: {value}")
            elif isinstance(value, dict):
                text_parts.append(self._extract_text_from_dict(value))
            elif isinstance(value, list):
                text_parts.append(self._extract_text_from_list(value))
            else:
                text_parts.append(f"{key}: {str(value)}")
        return "\n".join(text_parts)
    
    def _extract_text_from_list(self, data):
        """Extract text from list recursively."""
        text_parts = []
        for item in data:
            if isinstance(item, str):
                text_parts.append(item)
            elif isinstance(item, (int, float)):
                text_parts.append(str(item))
            elif isinstance(item, dict):
                text_parts.append(self._extract_text_from_dict(item))
            elif isinstance(item, list):
                text_parts.append(self._extract_text_from_list(item))
            else:
                text_parts.append(str(item))
        return "\n".join(text_parts)

class ParserFactory:
    """Factory for creating appropriate parsers based on file extension."""
    
    # Docling handles these formats
    _docling_formats = {'.pdf', '.docx', '.pptx', '.md'}
    
    # Specialized parsers
    _specialized_parsers = {
        '.txt': TxtParser,
        '.json': JSONParser,
        '.html': HTMLParser,
        '.htm': HTMLParser,
    }
    
    @classmethod
    def get_parser(cls, file_path):
        """Get appropriate parser for file extension."""
        ext = Path(file_path).suffix.lower()
        
        # Check specialized parsers first
        if ext in cls._specialized_parsers:
            parser_class = cls._specialized_parsers[ext]
            logger.info(f"Using {parser_class.__name__} for {ext} file")
            return parser_class()
        
        # Use Docling for supported formats
        elif ext in cls._docling_formats:
            logger.info(f"Using DoclingParser for {ext} file")
            return DoclingParser()
        
        else:
            logger.warning(f"No parser available for extension: {ext}")
            return None
    
    @classmethod
    def supported_extensions(cls):
        """Get list of supported file extensions."""
        return list(cls._specialized_parsers.keys()) + list(cls._docling_formats)