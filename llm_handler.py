"""
LLM Handler: Text generation with strict prompting for hallucination prevention.

Refactored for Project 2:
- LLMGenerator now implements ILLMGenerator  (OCP: swappable without changing RAGGenerator)
- RAGGenerator accepts ILLMGenerator via constructor injection  (DIP)
"""

from typing import Tuple

from interfaces import ILLMGenerator
from config import (
    LLM_MODEL, LLM_MAX_LENGTH, LLM_TEMPERATURE, LLM_TOP_P,
    RETRIEVAL_PROMPT_TEMPLATE, VERBOSE_LOGGING,
)


class LLMGenerator(ILLMGenerator):
    """
    HuggingFace transformers implementation of ILLMGenerator.
    Uses GPT-2 by default; any model name may be passed in.
    """

    def __init__(self, model_name: str = LLM_MODEL):
        from transformers import pipeline  # lazy import — keeps tests fast
        self.model_name = model_name
        print(f"Loading LLM model: {model_name}")
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            device=-1,
        )
        print("  -> Model loaded and ready for generation")

    def generate(
        self,
        prompt: str,
        max_length: int = LLM_MAX_LENGTH,
        temperature: float = LLM_TEMPERATURE,
        top_p: float = LLM_TOP_P,
        **kwargs,
    ) -> str:
        from transformers import GenerationConfig, set_seed  # lazy import
        set_seed(42)
        generation_config = GenerationConfig(
            max_new_tokens=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=self.generator.tokenizer.eos_token_id,
        )
        output = self.generator(
            prompt,
            generation_config=generation_config,
            return_full_text=True,
        )
        generated_text = output[0]["generated_text"]
        return generated_text[len(prompt):].strip()


class PromptBuilder:
    """Constructs prompts that enforce strict context-only answers."""

    @staticmethod
    def build_retrieval_prompt(context: str, query: str) -> str:
        return RETRIEVAL_PROMPT_TEMPLATE.format(context=context, query=query)


class AnswerProcessor:
    """Post-processes generated answers: cleans artifacts and assesses confidence."""

    UNCERTAINTY_PHRASES = [
        "i don't know",
        "i cannot find",
        "i'm not sure",
        "i don't have",
        "it's unclear",
        "i don't see",
        "i'm unable to",
        "i cannot determine",
    ]

    @staticmethod
    def contains_uncertainty(text: str) -> bool:
        text_lower = text.lower()
        return any(phrase in text_lower for phrase in AnswerProcessor.UNCERTAINTY_PHRASES)

    @staticmethod
    def clean_answer(text: str) -> str:
        if text.endswith(",") or text.endswith(";"):
            text = text.rstrip(",;")
        lines = text.split("\n")
        unique_lines = []
        for line in lines:
            if line not in unique_lines:
                unique_lines.append(line)
        return "\n".join(unique_lines).strip()

    @staticmethod
    def process(answer: str) -> Tuple[str, bool]:
        cleaned = AnswerProcessor.clean_answer(answer)
        has_uncertainty = AnswerProcessor.contains_uncertainty(cleaned)
        is_confident = not has_uncertainty and len(cleaned) > 20
        return cleaned, is_confident


class RAGGenerator:
    """
    Complete generation pipeline: retrieved context + query -> answer.

    DIP: depends on ILLMGenerator (abstraction), not LLMGenerator (concrete).
    OCP: swap in any ILLMGenerator (e.g. OpenAIGenerator, MockLLMGenerator)
         without touching this class.
    """

    def __init__(self, llm: ILLMGenerator | None = None):
        self.llm: ILLMGenerator = llm or LLMGenerator()
        self.prompt_builder = PromptBuilder()

    def generate_answer(
        self,
        context: str,
        query: str,
        citations: list,
    ) -> Tuple[str, list, bool]:
        if not context or not context.strip():
            return (
                "I cannot find information to answer your question in the available documentation.",
                [],
                False,
            )

        prompt = self.prompt_builder.build_retrieval_prompt(context, query)

        if VERBOSE_LOGGING:
            print(f"Generated prompt (first 200 chars): {prompt[:200]}...")

        answer = self.llm.generate(prompt)
        cleaned_answer, is_confident = AnswerProcessor.process(answer)

        if len(cleaned_answer) < 20:
            cleaned_answer = "I could not generate a clear answer from the available documentation."
            is_confident = False

        return cleaned_answer, citations, is_confident


def create_generator(llm: ILLMGenerator | None = None) -> RAGGenerator:
    """Factory: create a RAGGenerator, optionally injecting a custom ILLMGenerator."""
    return RAGGenerator(llm=llm)
