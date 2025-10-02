class RetrieverStub:
    def retrieve(self, query):
        return ["Documento exemplo sobre RAG"]


class GeneratorStub:
    def generate(self, context, question):
        return "Resposta gerada pelo LLM (stub)"


def main():
    retriever = RetrieverStub()
    generator = GeneratorStub()
    question = "O que é Retrieval-Augmented Generation?"
    context = retriever.retrieve(question)
    answer = generator.generate(context, question)
    print("Pergunta:", question)
    print("Contexto:", context)
    print("Resposta:", answer)


if __name__ == "__main__":
    main()
