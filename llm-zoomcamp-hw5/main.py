from starter import read_external_data, build_index, create_rag, calculate_cost
from sqllite_exporter import SQLiteSpanExporter
from rag_helper import RAGBase

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd


class RAGTraced(RAGBase):
  def __init__(self, index, llm_client):
    super().__init__(index, llm_client)
    self.tracer = trace.get_tracer("llm-zoomcamp")

  def rag(self, query):
    with self.tracer.start_as_current_span("rag") as span:
      return super().rag(query)

  def llm(self, prompt):
    with self.tracer.start_as_current_span("llm") as span:
      response = super().llm(prompt)
      usage = response.usage
      span.set_attribute("input_tokens", usage.input_tokens)
      span.set_attribute("output_tokens", usage.output_tokens)
      span.set_attribute("cost", calculate_cost(self.model, usage))
      return response

  def search(self, query):
    with self.tracer.start_as_current_span("search") as span:
      return super().search(query)

def calculate_token_stability(rows):
  df = pd.DataFrame(rows, columns=["name", "input_tokens", "output_tokens"])
  df = df[df["name"] == "llm"]
  df["token_stability"] = (df["input_tokens"] + df["output_tokens"]) / 2
  return df["token_stability"].values.tolist()

def query_tokens(exporter):
  return exporter.query("SELECT name, input_tokens, output_tokens FROM spans")
  

def main(skip_rag=False):
  load_dotenv()
  documents = read_external_data()
  index = build_index(documents)
  # rag = create_rag(index)

  provider = TracerProvider()
  # provider.add_span_processor(
  #     SimpleSpanProcessor(ConsoleSpanExporter())
  # )
  sqlite_exporter = SQLiteSpanExporter("traces.db")
  provider.add_span_processor(
    SimpleSpanProcessor(sqlite_exporter)
  )
  trace.set_tracer_provider(provider)

  # tracer = trace.get_tracer("llm-zoomcamp")

  # with tracer.start_as_current_span("llm-zoomcamp-hw5") as span:
  #   query = "How does the agentic loop keep calling the model until it stops?"
  #   answer = rag.rag(query)
  #   print(answer)

  if not skip_rag:
    client = OpenAI()
    rag_traced = RAGTraced(index, client)

    query = "How does the agentic loop keep calling the model until it stops?"
    answer = rag_traced.rag(query)
    print(answer)
  

  # calculate token stability
  token_stability = calculate_token_stability(query_tokens(sqlite_exporter))
  print(f"Token stability: {token_stability}")

if __name__ == "__main__":
  main(skip_rag=True)