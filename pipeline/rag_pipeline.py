from retriever.retriever import RetrieverStub
from generator.generator import GeneratorStub


class RAGPipeline:
    def __init__(self):
        self.retriever = RetrieverStub()
        self.generator = GeneratorStub()

    def run(self, question):
        context = self.retriever.retrieve(question)
        answer = self.generator.generate(context, question)
        return context, answer
