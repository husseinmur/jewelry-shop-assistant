import streamlit as st
from shared.config import init_apis

st.write("Testing API connections...")

try:
    openai_client, pinecone_index = init_apis()
    st.success("✅ APIs connected successfully!")
    st.write(f"Pinecone index: {pinecone_index}")
except Exception as e:
    st.error(f"❌ Setup error: {e}")
