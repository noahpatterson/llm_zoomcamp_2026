"""Starter code for the monitoring homework.

Sets up the text-search RAG from homework 1 and a shared OpenAI client.
"""

from openai import OpenAI

from gitsource import GithubRepositoryDataReader
from minsearch import Index

from rag_helper import RAGBase

from dotenv import load_dotenv
load_dotenv()

COMMIT = "8c1834d"

# --- Load the course lessons (same as HW1, HW2, HW4) ---

def read_external_data():
  reader = GithubRepositoryDataReader(
      repo_owner="DataTalksClub",
      repo_name="llm-zoomcamp",
      commit_id=COMMIT,
      allowed_extensions={"md"},
      filename_filter=lambda path: "/lessons/" in path,
  )
  documents = [file.parse() for file in reader.read()]
  return documents

def build_index(documents):
  index = Index(text_fields=["content"], keyword_fields=["filename"])
  index.fit(documents)
  return index

def create_rag(index, client):
  rag = RAGBase(index=index, llm_client=client)
  return rag

def calculate_cost(model, usage):
  cost = 0
  if model == "gpt-5.4-mini" in model:
    cost = ((usage.input_tokens * 0.15) + (usage.output_tokens * 0.60)) / 1_000_000
  return cost

# if __name__ == "__main__":
#     documents = read_external_data()
#     index = build_index(documents)
#     rag = create_rag(index)
#     query = "How does the agentic loop keep calling the model until it stops?"
#     answer = rag.rag(query)
#     print(answer)
