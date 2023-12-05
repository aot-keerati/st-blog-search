import openai
import pinecone
import streamlit as st

# OpenAI API key
openai.api_key = "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"  ### TN Aot Key | private using ### #ntkkey

# get the Pinecone API key and environment
pinecone.init(api_key="0a310604-6e64-4616-b258-a049bec92e82", ### TN Aot Key | private using ###
                    environment="gcp-starter")

index_name = "blog-index"
index = pinecone.Index(index_name = index_name)

def related_articles(query, k):
    # index = pinecone.Index(index_name = index_name)

    query_vector = openai.embeddings.create(
                    input = query,
                    model = "text-embedding-ada-002"
                ).data[0].embedding

    search_response = index.query(
        top_k = k,
        vector = query_vector,
        include_metadata = True
    )

    return search_response['matches']

prompt = """You're speaking with a user who needs an answer or description in question form.
Your goal is to provide helpful information based on your knowledge from the 'related articles', giving a confident and informative answer.

If you don't have the necessary information, let them know.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

These are relevant excerpts from the retrieved articles
(some may not be directly related to the query.
Please consider the most relevant information for your response.): {related_articles}

Please provide your response to the following question in the same language as the context:
    query: {user_query}
"""

def qa(query, articles):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            { "role": "system", "content": "You are a polite assistant" },
            { "role": "user", "content": prompt.format(
                related_articles = articles,
                user_query = query) }
        ],
        temperature=0, max_tokens=500
    )
    return response

### Front

st.write("Blog search")

query = st.text_input("search...")

if st.button('Search'):
    articles = related_articles(query, k=4)
    res = qa(query, articles)
    st.write(res.choices[0].message.content)
    st.write(res)
    st.write(articles)

e = RuntimeError('This is an exception of type RuntimeError')
st.exception(e)
