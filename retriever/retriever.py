from logging.logger import get_logger

logger = get_logger("retriever")


class RetrieverStub:
    """
    Stub retriever that returns a static example document.
    """

    def retrieve(self, query):
        logger.info(f"Retrieving context for query: {query}")
        return ["Sample document about RAG"]
