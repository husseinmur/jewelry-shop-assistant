#!/usr/bin/env python3
"""
Quick test to verify dimension fix
"""

import os
import sys
sys.path.append('.')

# Mock streamlit for testing
class MockSecrets:
    def __getitem__(self, key):
        return os.getenv(key)

class MockStreamlit:
    secrets = MockSecrets()

    @staticmethod
    def error(msg):
        print(f"ERROR: {msg}")

    @staticmethod
    def success(msg):
        print(f"SUCCESS: {msg}")

    @staticmethod
    def info(msg):
        print(f"INFO: {msg}")

    @staticmethod
    def warning(msg):
        print(f"WARNING: {msg}")

sys.modules['streamlit'] = MockStreamlit()

try:
    from shared.config import init_apis
    from shared.langchain_rag import init_langchain_rag

    print("🔧 Initializing APIs...")
    openai_client, pinecone_index = init_apis()
    print("✅ APIs connected")

    print("🚀 Testing LangChain RAG...")
    rag_system = init_langchain_rag(pinecone_index, os.getenv("OPENAI_API_KEY"))

    if rag_system:
        print("✅ LangChain RAG initialized successfully")

        # Test a simple query
        print("\n🔍 Testing query: 'عندكن سلاسل'")
        response, results = rag_system.conversational_search("عندكن سلاسل")

        print(f"💬 Response: {response[:100]}...")
        print(f"📊 Found {len(results) if results else 0} results")

    else:
        print("❌ Failed to initialize RAG system")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()