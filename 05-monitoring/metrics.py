import time
from dataclasses import dataclass, field
from rag_helper import RAGBase

@dataclass
class LLMCallRecord:
  model: str
  prompt: str
  instructions: str
  question: str
  answer: str
  prompt_tokens: int
  completion_tokens: int
  total_tokens: int
  response_time: float
  cost: float
  timestamp: float = field(default_factory=time.time)

def calculate_cost(model, usage):
  cost = 0
  if model == "gpt-5.4-mini" in model:
    cost = ((usage.input_tokens * 0.15) + (usage.output_tokens * 0.60)) / 1_000_000
  return cost

class RAGWithMetrics(RAGBase):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.last_call_record: LLMCallRecord = None
    self.logs: list[LLMCallRecord] = []

  def rag(self, query):
    self._current_question = query
    return super().rag(query)

  def llm(self, prompt: str) -> str:
    start_time = time.time()
    response = self._call_llm(prompt)
    response_time = time.time() - start_time
    question = getattr(self, "_current_question", "")
    self.log_response(question, prompt, response, response_time)
    return response.output_text

  def _call_llm(self, prompt):
    input_messages = [
      {'role': 'developer', 'content': self.instructions},
      {'role': 'user', 'content': prompt},
    ]
    response = self.llm_client.responses.create(
      model=self.model,
      input=input_messages,
    )
    return response

  def log_response(self, question: str, prompt: str, response: str, response_time: float):
    usage = response.usage
    cost = calculate_cost(self.model, usage)
    call_record = LLMCallRecord(
      model=self.model,
      question=question,
      prompt=prompt,
      instructions=self.instructions,
      answer=response.output_text,
      prompt_tokens=usage.input_tokens,
      completion_tokens=usage.output_tokens,
      total_tokens=usage.total_tokens,
      response_time=response_time,
      cost=cost
    )
    self.logs.append(call_record)
    self.last_call_record = call_record