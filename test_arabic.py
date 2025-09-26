# test_arabic.py
import streamlit as st
from shared.config import init_apis
from shared.embeddings import expand_search_query

st.title("اختبار البحث العربي")

openai_client, pinecone_index = init_apis()

test_query = st.text_input("استعلام البحث:", value="مجوهرات بشكل الياسمين")

if st.button("اختبار توسيع الاستعلام"):
    if test_query:
        expansion = expand_search_query(test_query)
        st.write("**توسيع الاستعلام:**")
        st.write(expansion)
