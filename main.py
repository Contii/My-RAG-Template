import time
import warnings
from pipeline.rag_pipeline import RAGPipeline
from logging.logger import setup_logger, get_logger


warnings.filterwarnings("ignore", category=UserWarning)

setup_logger()
logger = get_logger("main")

def main():
    logger.info("Starting My-RAG-Template")
    print("\n-------My-RAG-Template-------")
    print("1 - Direct question to LLM.")
    print("2 - Feed prompt with RAG.")
    mode = input("Choose mode: ").strip()
    use_rag = True if mode == "2" else False

    logger.info(f"Loading LLM model in mode {mode}...")
    print("\nLoading LLM model...")
    start_load = time.time()
    pipeline = RAGPipeline(use_rag=use_rag)
    end_load = time.time()
    logger.info(f"LLM loaded in {end_load - start_load:.2f} seconds.")
    print(f"LLM loaded in {end_load - start_load:.2f} seconds.")

    print("Type your questions below (type 'exit' to quit):")
    print("--------------------------------------------------")
    while True:
        question = input("\nQuestion: ")
        logger.info(f"Received question: {question}")
        if question.strip().lower() == "exit":
            logger.info("Exiting application.")
            print("Exiting...")
            break
        start_gen = time.time()
        context, answer = pipeline.run(question)
        end_gen = time.time()
        if use_rag:
            logger.info(f"Received context: {context}")
            print("Context:", context)
        logger.info(f"Received answer: {answer}")
        print("\nAnswer:", answer)
        logger.info(f"Answer generated in {end_gen - start_gen:.2f} seconds.")
        print(f"\nAnswer generated in {end_gen - start_gen:.2f} seconds.")
        print("--------------------------------------------------")


if __name__ == "__main__":
    main()
