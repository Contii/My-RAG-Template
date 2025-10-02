import unittest
from pipeline.rag_pipeline import RAGPipeline


class TestRAGPipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = RAGPipeline()

    def test_run_returns_context_and_answer(self):
        question = "What is Retrieval-Augmented Generation?"
        context, answer = self.pipeline.run(question)
        self.assertIsInstance(context, list)
        self.assertTrue(len(context) > 0)
        self.assertIsInstance(answer, str)
        self.assertIn("LLM", answer)


if __name__ == "__main__":
    unittest.main()
