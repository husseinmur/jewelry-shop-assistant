#!/usr/bin/env python3
"""
Test script to verify semantic search improvements
"""

import streamlit as st
from shared.config import init_apis
from shared.database import smart_search

def test_search_queries():
    """Test various search queries to verify semantic matching"""

    # Initialize APIs
    try:
        openai_client, pinecone_index = init_apis()
        print("✅ Connected to APIs successfully")
    except Exception as e:
        print(f"❌ Failed to connect to APIs: {e}")
        return

    # Test queries that should work with semantic search
    test_queries = [
        # Descriptive queries (should find semantically similar items)
        "مجوهرات ذهبية بلمعة جميلة",
        "قطعة أنيقة للمناسبات الخاصة",
        "شيء بسيط للاستخدام اليومي",
        "قطعة فاخرة للزفاف",
        "مجوهرات بتصميم عصري",
        "شيء كلاسيكي وأنيق",

        # Mixed queries (keyword + description)
        "خاتم بسيط وأنيق",
        "عقد ذهبي فاخر",
        "أقراط للمناسبات",

        # Pure keywords (should still work)
        "خاتم",
        "عقد",
        "أقراط"
    ]

    print("\n🔍 Testing semantic search queries:")
    print("=" * 50)

    for query in test_queries:
        print(f"\n🔎 Query: '{query}'")
        try:
            results = smart_search(pinecone_index, query, search_type="text", top_k=3)

            if results:
                print(f"✅ Found {len(results)} results:")
                for i, result in enumerate(results, 1):
                    metadata = result.metadata
                    score = result.score
                    print(f"  {i}. {metadata.get('name', 'Unknown')} (Score: {score:.3f})")
                    print(f"     Category: {metadata.get('category', 'N/A')}")
                    print(f"     Price: {metadata.get('price', 'N/A')} ريال")
            else:
                print("❌ No results found")

        except Exception as e:
            print(f"❌ Error: {e}")

        print("-" * 30)

if __name__ == "__main__":
    print("🧪 Semantic Search Test")
    print("Testing improved vector search functionality...")
    test_search_queries()
    print("\n✨ Test completed!")