import logging

class RetrieverStub:
    """
    Stub retriever that returns a static example document.
    """

    def retrieve(self, query):
        logging.info(f"Retrieving context for query: {query}")
        return ["Sample document about RAG"]
