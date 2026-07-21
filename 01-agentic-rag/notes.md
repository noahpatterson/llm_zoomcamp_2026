# Notes

## What is RAG

- Retrieval augmented generation
- Is the most popular used in the industry.
- R - retrieval - query to the knowledgebase/database based on the question
- A - augment - build prompt with knowledge
- G - generate - send to llm

### search

- You probably don't want to send massive amount of search data directly to the llm. Using a tool to index and search
  will probably be faster. Though, I wonder if tokenizing the data and sending it would help?
- minsearch - is a lightweight way of searching datasets for realtively small datasets
  - text_fields - are where the answer might be found
  - keyword - is something you need an exact match for. used to restrict the search space to a specific sub-space.
  - boosting - you can say that one field will be more important than another
