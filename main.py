from pipeline.rag_pipeline import RAGPipeline


def main():
    pipeline = RAGPipeline()
    question = "What is Retrieval-Augmented Generation?"
    context, answer = pipeline.run(question)
    print("Question:", question)
    print("Context:", context)
    print("Answer:", answer)


if __name__ == "__main__":
    main()
