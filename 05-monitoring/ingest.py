import requests
from minsearch import Index
from sqlitesearch import TextSearchIndex


def load_faq_data():
  docs_url = 'https://datatalks.club/faq/json/courses.json'
  try:
      response = requests.get(docs_url)
      courses_raw = response.json()
      print(courses_raw)
  except Exception as e:
      print(f'Error: {e}')


  documents = []
  url_prefix = 'https://datatalks.club/faq/'

  for course in courses_raw:
      course_url = url_prefix + course['path']
      course_response = requests.get(course_url)
      course_response.raise_for_status()
      course_data = course_response.json()
      documents.extend(course_data)
  
  return documents

def build_index(documents):
  index = Index(
    text_fields=['question', 'section', 'answer'],
    keyword_fields=['course']
  )

  index.fit(documents)
  return index