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
    llm_client, 
    instructions=INSTRUCTIONS, 
    user_prompt_template=USER_PROMPT_TEMPLATE,
    course='llm-zoomcamp',
    model='gpt-5.4',
    reasoning='medium'
  ):
    self.index = index
    self.llm_client = llm_client
    self.instructions = instructions
    self.user_prompt_template = user_prompt_template
    self.course = course
    self.model = model
    self.reasoning = reasoning

  def search(self, query, num_results=5):
    boost_dict={'question': 2.0}
    filter_dict={'course': self.course}

    return self.index.search(
      query, 
      boost_dict=boost_dict,
      filter_dict=filter_dict,
      num_results=num_results
  )

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

  def llm(self, prompt):
      message_history = [
        { 'role': 'developer', 'content': self.instructions },
        { 'role': 'user', 'content': prompt }
      ]
      response = self.llm_client.responses.create(
          model=self.model,
          reasoning={"effort": self.reasoning},
          input=message_history
      )
      return response

  def rag(self, query):
    search_results = self.search(query)
    prompt = self.build_prompt(query, search_results)
    answer = self.llm(prompt)
    return answer.output_text  

class OllamaRAGHelper(RAGHelper):
  def llm(self, prompt):
    return self.llm_client.chat.completions.create(
      model=self.model,
      messages=[
        {"role": "developer", "content": self.instructions},
        {"role": "user", "content": prompt}
      ]
    )