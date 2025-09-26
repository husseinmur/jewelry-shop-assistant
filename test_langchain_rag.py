#!/usr/bin/env python3
"""
Test LangChain RAG implementation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from shared.config import init_apis
from shared.langchain_rag import init_langchain_rag

def test_langchain_rag():
    """Test LangChain RAG system with various Arabic queries"""

    print("🧪 Testing LangChain RAG System")
    print("=" * 50)

    # Initialize
    try:
        print("🔧 Initializing APIs...")
        openai_client, pinecone_index = init_apis()
        print("✅ APIs connected")

        print("🚀 Initializing LangChain RAG...")
        rag_system = init_langchain_rag(pinecone_index, os.getenv("OPENAI_API_KEY"))
        if not rag_system:
            print("❌ Failed to initialize RAG system")
            return

        print("✅ LangChain RAG initialized")

    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        return

    # Test queries
    test_queries = [
        # Natural language queries
        "عندكن سلاسل ذهبية؟",
        "أريد شيء أنيق للزفاف",
        "ما المتوفر في الخواتم؟",
        "قطعة بسيطة للاستخدام اليومي",
        "مجوهرات فاخرة للمناسبات",

        # Descriptive queries
        "شيء عصري ومميز",
        "قطعة كلاسيكية وأنيقة",
        "مجوهرات بتصميم بسيط",

        # Material-based
        "قطع ذهبية",
        "مجوهرات فضية",

        # Traditional keywords
        "خاتم",
        "عقد",
        "أقراط"
    ]

    print("\n🔍 Testing Natural Language Queries:")
    print("-" * 50)

    for i, query in enumerate(test_queries, 1):
        print(f"\n🔎 Test {i}: '{query}'")
        try:
            # Test conversational search
            response, results = rag_system.conversational_search(query)

            print(f"💬 Response: {response[:100]}...")

            if results:
                print(f"✅ Found {len(results)} products:")
                for j, result in enumerate(results[:3], 1):  # Show top 3
                    metadata = result['metadata']
                    print(f"  {j}. {metadata.get('name', 'Unknown')} - {metadata.get('price', 'N/A')} ريال")
            else:
                print("❌ No products found")

        except Exception as e:
            print(f"❌ Error: {e}")

        print("-" * 30)

    print("\n✨ LangChain RAG Testing Completed!")

if __name__ == "__main__":
    # Set up environment
    import streamlit as st

    # Mock streamlit secrets for testing
    if not hasattr(st, 'secrets'):
        class MockSecrets:
            def __getitem__(self, key):
                return os.getenv(key)
        st.secrets = MockSecrets()

    test_langchain_rag()