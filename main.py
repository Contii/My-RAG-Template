import time
import warnings
from datetime import datetime, timedelta
from pipeline.rag_pipeline import RAGPipeline
from logger.logger import setup_logger, get_logger, log_model_loading_metrics, close_metrics_session

warnings.filterwarnings("ignore", category=UserWarning)

setup_logger()
logger = get_logger("main")

def get_filter_options(pipeline):
    """Get filter options from user."""
    if not hasattr(pipeline.retriever, 'get_available_file_types'):
        return None
    
    print("\n============ FILTER OPTIONS ============")
    available_types = pipeline.retriever.get_available_file_types()
    available_sources = pipeline.retriever.get_available_sources()
    
    print(f"Available file types: {available_types}")
    print(f"Available sources: {available_sources}")
    
    use_filters = input("Use filters? (y/n): ").strip().lower()
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
    
    # Date filters (optional - most documents don't have dates)
    use_dates = input("\nFilter by dates? (y/n): ").strip().lower()
    if use_dates == 'y':
        from datetime import datetime
        
        date_from_input = input("From date (YYYY-MM-DD, or press Enter to skip): ").strip()
        if date_from_input:
            try:
                filters['date_from'] = datetime.fromisoformat(date_from_input)
            except ValueError:
                print("Invalid date format, skipping from date")
        
        date_to_input = input("To date (YYYY-MM-DD, or press Enter to skip): ").strip()
        if date_to_input:
            try:
                filters['date_to'] = datetime.fromisoformat(date_to_input)
            except ValueError:
                print("Invalid date format, skipping to date")
    
    return filters if filters else None

def main():
    logger.info("Starting My-RAG-Template")
    print("\n=========== My-RAG-Template ===========")
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

    print("="*40)
    print("\nType your questions below (type 'exit' to quit, 'metrics' to see dashboard):")
 
    
    while True:
        question = input("Question: ")
        
        if question.strip().lower() == "exit":
            # Save final metrics report before exit
            if hasattr(pipeline.retriever, 'save_metrics_report'):
                pipeline.retriever.save_metrics_report()
                print("📊 Metrics report saved to logs/retrieval_report.txt")
            
            close_metrics_session()
            logger.info("Exiting application.")
            print("Exiting...")
            break
        
        if question.strip().lower() == "metrics":
            if hasattr(pipeline.retriever, 'print_metrics_dashboard'):
                pipeline.retriever.print_metrics_dashboard()
                
                insights = pipeline.retriever.get_performance_insights()
                if insights and any('tracking is disabled' not in insight for insight in insights):
                    print("\n💡 PERFORMANCE INSIGHTS:")
                    for insight in insights:
                        print(f"   {insight}")
            else:
                print("Metrics not available for this retriever type")
            continue

        logger.info(f"Received question: {question}")
        
        # Get filters if in filter mode
        filters = None
        if use_filters:
            filters = get_filter_options(pipeline)
        
        start_gen = time.time()
        context, answer = pipeline.run(question, filters)
        end_gen = time.time()
        
        # Display results in organized format
        if use_rag and context:
            print("\n" + "="*40)
            print("    📄 RETRIEVED CONTEXT")
            print("="*40)
            for i, ctx in enumerate(context, 1):
                print(f"\n📋 Document {i}:")
                print(f"{ctx}")
            print("="*40)
        
        # Extract and display clean answer
        final_answer = answer
        
        # If answer contains the full prompt structure, extract just the final answer
        if "Answer:" in answer:
            answer_parts = answer.split("Answer:")
            if len(answer_parts) > 1:
                final_answer = answer_parts[-1].strip()
                
                # Remove any trailing numbers or artifacts
                lines = final_answer.split('\n')
                clean_lines = []
                for line in lines:
                    line = line.strip()
                    # Skip lines that are just numbers or artifacts
                    if line and not all(c in '0. ' for c in line):
                        clean_lines.append(line)
                        break  # Take only the first clean line as the answer
                
                final_answer = clean_lines[0] if clean_lines else final_answer
        
        print(f"\n❓ Question: {question}")
        print(f"\n🤖 Answer: {final_answer}")
        print(f"\n⏱️ Generation time: {end_gen - start_gen:.2f} seconds")
        print("="*40,"\n")

if __name__ == "__main__":
    main()
