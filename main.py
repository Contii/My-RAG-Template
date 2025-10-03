import time
import warnings
from pipeline.rag_pipeline import RAGPipeline

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    print("\n-------My-RAG-Template-------")
    print("1 - Direct question to LLM.")
    print("2 - Feed prompt with RAG.")
    mode = input("Choose mode: ").strip()
    use_rag = True if mode == "2" else False

    print("\nLoading LLM model...")
    start_load = time.time()
    pipeline = RAGPipeline(use_rag=use_rag)
    end_load = time.time()
    print(f"LLM loaded in {end_load - start_load:.2f} seconds.")

    print("Type your questions below (type 'exit' to quit):")
    print("--------------------------------------------------")
    while True:
        question = input("\nQuestion: ")
        if question.strip().lower() == "exit":
            print("Exiting...")
            break
        start_gen = time.time()
        context, answer = pipeline.run(question)
        end_gen = time.time()
        if use_rag:
            print("Context:", context)
        print("\nAnswer:", answer)
        print(f"\nAnswer generated in {end_gen - start_gen:.2f} seconds.")
        print("--------------------------------------------------")


if __name__ == "__main__":
    main()
