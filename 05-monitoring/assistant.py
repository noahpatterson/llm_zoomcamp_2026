import sys
import os

from metrics import RAGWithMetrics
from ingest import load_faq_data, build_index
from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
from db_save import save_conversation

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

def create_assistant():
  documents = load_faq_data()
  index = build_index(documents)

  return RAGWithMetrics(
    index=index,
    llm_client=OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
  )

if __name__ == "__main__":
  assistant = create_assistant()
  
  query = "How do I join the course?"
  if len(sys.argv) > 1:
    query = sys.argv[1]

  response = assistant.rag(query)
  save_conversation(assistant.last_call_record, query, "llm-zoomcamp")
  print(response)