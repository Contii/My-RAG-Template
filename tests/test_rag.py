import unittest
from generator.generator import GeneratorStub, LLMGenerator


class TestGenerator(unittest.TestCase):
    def test_stub_generator(self):
        generator = GeneratorStub()
        context = ["Sample context"]
        question = "What is RAG?"
        answer = generator.generate(context, question)
        self.assertIsInstance(answer, str)
        self.assertIn("LLM", answer)

    def test_llm_generator_init(self):
        # Test model loading error handling (use an invalid model name)
        with self.assertRaises(RuntimeError):
            LLMGenerator("invalid-model-name")

    def test_llm_generator_generate(self):
        # This test will only run if the model is available and downloaded
        try:
            generator = LLMGenerator("microsoft/bitnet-b1.58-2B-4T", max_tokens=10)
            context = ["Sample context"]
            question = "What is RAG?"
            answer = generator.generate(context, question)
            self.assertIsInstance(answer, str)
        except RuntimeError:
            self.skipTest("LLM model not available for testing.")


if __name__ == "__main__":
    unittest.main()
