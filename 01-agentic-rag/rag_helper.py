import json
from openai import OpenAI

INSTRUCTIONS = """
You are a helpful assistant that can answer questions about the course given the provided context.
Use the context to find relevant information and provide accurate answers.
If an answer is not found in the context, respond with "I don't know"
"""

USER_PROMPT_TEMPLATE = """
Question: {user_question}

Context:
{context}
"""

class RAGHelper:
  def __init__(
    self, 
    index, 
    llm_client: OpenAI, 
    instructions=INSTRUCTIONS, 
    user_prompt_template=USER_PROMPT_TEMPLATE,
    tools: list[dict] = [],
    course='llm-zoomcamp',
    model='gpt-5.4',
    reasoning='medium'
  ):
    self.index = index
    self.llm_client = llm_client
    self.instructions = instructions
    self.user_prompt_template = user_prompt_template
    self.tools = tools
    self.course = course
    self.model = model
    self.reasoning = reasoning

  def search(self, query, num_results=5):
    boost_dict={'question': 2.0}
    filter_dict={'course': self.course}

    results = self.index.search(
      query, 
      boost_dict=boost_dict,
      filter_dict=filter_dict,
      num_results=num_results
    )
    print(f'Found {len(results)} results')
    return results

  def build_context(self, search_results):
    lines = []

    for doc in search_results:
      lines.append(doc["section"])
      lines.append(f'Q: {doc["question"]}')
      lines.append(f'A: {doc["answer"]}')
      lines.append('')

    return '\n'.join(lines)

  def build_prompt(self, query, search_results):
    context = self.build_context(search_results)
    prompt = self.user_prompt_template.format(
      user_question=query, 
      context=context
    )
    return prompt.strip()

  def make_call(self, call):
    args = json.loads(call.arguments)
    
    if call.name == 'search':
      result = self.search(**args)

    result_json = json.dumps(result)

    return {
      "type": "function_call_output",
      "call_id": call.call_id,
      "output": result_json,
    }

  def llm(self, prompt):
      message_history = [
        { 'role': 'developer', 'content': self.instructions },
        { 'role': 'user', 'content': prompt }
      ]
      response = self.llm_client.responses.create(
          model=self.model,
          reasoning={"effort": self.reasoning},
          input=message_history,
      )
      return response

  def llm_with_tools(self, message_history):
    response = self.llm_client.responses.create(
      model=self.model,
      reasoning={"effort": self.reasoning},
      input=message_history,
      tools=self.tools
    )
    return response

  def rag(self, query):
    search_results = self.search(query)
    prompt = self.build_prompt(query, search_results)
    answer = self.llm(prompt)
    return answer.output_text  

  def agentic_rag(self, query):
    
    if self.tools:
      message_history = [
            { 'role': 'developer', 'content': self.instructions },
            { 'role': 'user', 'content': query }
          ]
      iterations = 1
      
      while True:
        has_function_calls = False
        print(f'Iteration {iterations}')

        response = self.llm_with_tools(message_history)
        message_history.extend(response.output)

        for item in response.output:
          if item.type == "function_call":
              print("function_call:", item.name, item.arguments)
              call_output = self.make_call(item)
              message_history.append(call_output)
              has_function_calls = True

          elif item.type == "message":
              print("ASSISTANT:")
              last_answer = item.content[0].text
              print(last_answer)

        iterations = iterations + 1

        if not has_function_calls:
          break

      return last_answer

    else:
      return self.rag(query)

class OllamaRAGHelper(RAGHelper):
  def llm(self, prompt):
    return self.llm_client.chat.completions.create(
      model=self.model,
      messages=[
        {"role": "developer", "content": self.instructions},
        {"role": "user", "content": prompt}
      ]
    )