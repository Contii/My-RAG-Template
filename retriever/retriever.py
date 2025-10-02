class RetrieverStub:
    """
    Stub retriever that returns a static example document.
    """

    def retrieve(self, query):
        return ["Sample document about RAG"]
