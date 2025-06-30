import os,streamlit as st,pickle
from dotenv import load_dotenv
from utils.loader import load_docs
from utils.summarizer import summarize_docs
from utils.vector_store import build_index,load_index
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv('.config')
st.set_page_config(page_title='Scheme Research Tool')
if 'db' not in st.session_state:
    st.session_state['db']=None
if 'summary' not in st.session_state:
    st.session_state['summary']=''
with st.sidebar:
    urls=st.text_area('Enter scheme URLs, one per line')
    if st.button('Process URLs'):
        all_docs=[]
        for url in [u.strip() for u in urls.splitlines() if u.strip()]:
            all_docs+=load_docs(url)
        if all_docs:
            st.session_state['summary']=summarize_docs(all_docs)
            db=build_index(all_docs,'faiss_store_openai.pkl')
            st.session_state['db']=db
if st.session_state['summary']:
    st.subheader('Summary')
    st.write(st.session_state['summary'])
query=st.text_input('Ask a question')
if query and st.session_state['db']:
    qa=RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0),chain_type='stuff',retriever=st.session_state['db'].as_retriever())
    answer=qa.run(query)
    st.write(answer)