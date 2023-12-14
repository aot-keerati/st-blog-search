import ast
import cohere
import openai
import pinecone
import streamlit as st

PINECONE_API_KEY = '0a310604-6e64-4616-b258-a049bec92e82'
PINECONE_ENV = 'gcp-starter'
COHERE_API_KEY = 'kpZfBhHE4FSzvdZeSE2SK50oblBd7kHYVLqZlGeJ'

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
co = cohere.Client(COHERE_API_KEY)
INDEX_NAME = 'blog-index'
index = pinecone.Index(index_name=INDEX_NAME)

# if INDEX_NAME not in pinecone.list_indexes():
#     # Index does not exist. Creating new index.
#     pinecone.create_index(INDEX_NAME, 768, metadata_config= {"indexed": ["url", "id"]})
# else:
#     # Index already exists. Deleting and Creating new index.
#     pinecone.delete_index(INDEX_NAME)
#     pinecone.create_index(INDEX_NAME, 768, metadata_config= {"indexed": ["url", "id"]})

f = open('dat/vectors-data.txt', 'r')
blog_vectors_import = f.read()
blog_vectors = ast.literal_eval(blog_vectors_import)
f.close()

f = open('dat/keep-vectors-data.txt', 'r')
keep_vectors_import = f.read()
keep_vectors = ast.literal_eval(keep_vectors_import)
f.close()

f = open('dat/prompt.txt', 'r')
PROMPT = f.read()
f.close()

def articles_search(query, k):
    query_vector = co.embed(texts = [query],
                                model='embed-multilingual-v2.0',
                                input_type='search_query').embeddings

    search_response = index.query(
        top_k = k,
        vector = query_vector,
        include_metadata = True
    )

    return search_response['matches']

def qa(query, articles):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{ "role": "user", "content": PROMPT.format(
                related_articles = articles,
                user_query = query) }
        ],
        temperature=0, max_tokens=500
    )
    return response

def create_hyperlink(url, text):
    return f'<a href="{url}" target="_blank">{text}</a>'

### Front
st.set_page_config(page_title="Blog search")
st.write("## Blog search")
st.caption("เว็บแอปพลิเคชั่นระบบสืบค้นข้อมูล ที่ค้นหาข้อมูลด้วย vectorized search และสร้างคำตอบด้วย GPT-3.5 model")

OPENAI_API_KEY = st.sidebar.text_input("กรอก OpenAI API Key:", type="password")

with st.form('form_1'):
  query = st.text_area('พิมพ์คำถามหรือคีย์เวิร์ดที่ต้องการค้นหา:', placeholder='เช่น ปลิงทะเลชมพู, พลังงานชีวภาพ คืออะไร')
  search_btn = st.form_submit_button('Submit')
  if search_btn and query:
      if not OPENAI_API_KEY:
        st.info("Please add your OpenAI API key to continue.")
      else:
        openai.api_key = OPENAI_API_KEY
        
        articles = articles_search(query, k=4)
        res = response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{ "role": "user", "content": PROMPT.format(
                    related_articles = articles,
                    user_query = query) }
            ],
            temperature=0, max_tokens=500
        )
    
        st.write('### Answer:')
        st.write(res.choices[0].message.content)
        st.write('บทความที่เกี่ยวข้องมากที่สุด ได้แก่:')
        st.markdown(create_hyperlink(articles[0]['metadata']['url'], articles[0]['metadata']['title']), unsafe_allow_html=True)
    
        st.caption(res.usage)
        st.write(articles)
