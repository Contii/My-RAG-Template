import time
import warnings
from datetime import datetime, timedelta
from pipeline.rag_pipeline import RAGPipeline
from logger.logger import setup_logger, get_logger, log_model_loading_metrics

warnings.filterwarnings("ignore", category=UserWarning)

setup_logger()
logger = get_logger("main")

def get_filter_options(pipeline):
    """Get filter options from user."""
    if not hasattr(pipeline.retriever, 'get_available_file_types'):
        return None
    
    print("\n=== FILTER OPTIONS ===")
    available_types = pipeline.retriever.get_available_file_types()
    available_sources = pipeline.retriever.get_available_sources()
    
    print(f"Available file types: {available_types}")
    print(f"Available sources: {available_sources}")
    
    use_filters = input("\nUse filters? (y/n): ").strip().lower()
    if use_filters != 'y':
        return None
    
    filters = {}
    
    # File type filter
    file_types_input = input(f"Filter by file types {available_types} (comma-separated, or press Enter to skip): ").strip()
    if file_types_input:
        filters['file_types'] = [ft.strip() for ft in file_types_input.split(',')]
    
    # Source filter
    sources_input = input(f"Filter by sources {available_sources} (comma-separated, or press Enter to skip): ").strip()
    if sources_input:
        filters['sources'] = [s.strip() for s in sources_input.split(',')]
    
    # Score filter
    min_score_input = input("Minimum score threshold (0.0-1.0, or press Enter for default): ").strip()
    if min_score_input:
        try:
            filters['min_score'] = float(min_score_input)
        except ValueError:
            print("Invalid score, using default")
    
    return filters if filters else None

def main():
    logger.info("Starting My-RAG-Template")
    print("\n-------My-RAG-Template-------")
    print("1 - Direct question to LLM.")
    print("2 - Feed prompt with RAG.")
    print("3 - RAG with advanced filters.")
    mode = input("Choose mode: ").strip()
    
    use_rag = mode in ["2", "3"]
    use_filters = mode == "3"

    logger.info(f"Loading LLM model in mode {mode}...")
    print("\nLoading LLM model...")
    start_load = time.time()
    pipeline = RAGPipeline(use_rag=use_rag)
    end_load = time.time()
    
    log_model_loading_metrics(logger, end_load - start_load)
    print(f"LLM loaded in {end_load - start_load:.2f} seconds.")

    print("Type your questions below (type 'exit' to quit):")
    print("--------------------------------------------------")
    while True:
        question = input("\nQuestion: ")
        if question.strip().lower() == "exit":
            logger.info("Exiting application.")
            print("Exiting...")
            break

        logger.info(f"Received question: {question}")
        
        # Get filters if in filter mode
        filters = None
        if use_filters:
            filters = get_filter_options(pipeline)
        
        start_gen = time.time()
        context, answer = pipeline.run(question, filters)
        end_gen = time.time()
        
        if use_rag:
            print("Context:", context)
        print("\nAnswer:", answer)
        print(f"\nAnswer generated in {end_gen - start_gen:.2f} seconds.")
        print("--------------------------------------------------")

if __name__ == "__main__":
    main()
