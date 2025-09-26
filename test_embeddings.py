import streamlit as st
from shared.config import init_apis
from shared.embeddings import expand_search_query, get_text_embedding

st.title("Test Embedding Functions")

# Initialize APIs
openai_client, pinecone_index = init_apis()

# Test query expansion
test_query = st.text_input("Test search query:", value="jasmine shaped jewelry")

if st.button("Test Query Expansion"):
    if test_query:
        expansion = expand_search_query(test_query)
        st.write("**Query Expansion:**")
        st.write(expansion)
        
        # Test embedding
        embedding = get_text_embedding(test_query)
        if embedding:
            st.success(f"✅ Embedding generated: {len(embedding)} dimensions")
        else:
            st.error("❌ Failed to generate embedding")
