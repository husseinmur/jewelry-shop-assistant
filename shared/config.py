import streamlit as st
import openai
from pinecone import Pinecone, ServerlessSpec

# Initialize APIs
def init_apis():
    """Initialize OpenAI and Pinecone clients"""
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index_name = st.secrets["PINECONE_INDEX_NAME"]
    
    # Create index if it doesn't exist
    try:
        index = pc.Index(index_name)
    except:
        # Create index with appropriate dimensions
        pc.create_index(
            name=index_name,
            dimension=1536,  # OpenAI embedding dimension
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        index = pc.Index(index_name)
    
    return openai, index

# Constants
EMBEDDING_MODEL = "text-embedding-ada-002"
VISION_MODEL = "gpt-4o"
TEXT_MODEL = "gpt-4"

# Image processing settings
MAX_IMAGE_SIZE = (800, 800)
THUMBNAIL_SIZE = (200, 200)
