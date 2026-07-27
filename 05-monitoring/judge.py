import json
import os
import random
from pathlib import Path

from pydantic import BaseModel
from typing import Literal
from openai import OpenAI
from dotenv import load_dotenv

from evaluation_utils import llm_structured_retry

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

class RelevanceVerdict(BaseModel):
    relevance: Literal["NON_RELEVANT", "PARTLY_RELEVANT", "RELEVANT"]
    explanation: str

judge_instructions = """
You are an expert evaluator for a RAG system.
Analyze the relevance of the generated answer to the given question.

Classify the answer as:
- RELEVANT: the answer addresses the question
- PARTLY_RELEVANT: the answer partially addresses the question
- NON_RELEVANT: the answer does not address the question
""".strip()

judge_prompt = """
Question: {question}
Generated Answer: {answer}
""".strip()

def evaluate_relevance(question, answer, client=None):
    if client is None:
        client = OpenAI()

    prompt = judge_prompt.format(
        question=question,
        answer=answer
    )

    result, usage = llm_structured_retry(
        client,
        judge_instructions,
        prompt,
        RelevanceVerdict,
    )

    return result.relevance, result.explanation


def get_judge_one_in_n():
    return int(os.getenv("JUDGE_ONE_IN_N", "10"))


def should_run_judge():
    one_in_n = get_judge_one_in_n()
    if one_in_n <= 0:
        return False
    if one_in_n == 1:
        return True
    return random.random() < (1 / one_in_n)


def print_relevance(relevance, explanation, streamlit=None):
    if streamlit is not None:
        streamlit.header("Judge Result")
        streamlit.write(f"Relevance: {relevance}")
        streamlit.write(f"Explanation: {explanation}")

if __name__ == "__main__":
    load_dotenv()

    question = "Can I still join the course?"
    answer = "Yes, you can still join. The course is self-paced."

    relevance, explanation = evaluate_relevance(question, answer)
    print(relevance)
    print(explanation)