#!/usr/bin/env python3
"""
Test the category-based fallback system
"""

import sys
sys.path.append('/home/hussein/shop-assistant')

from shared.config import init_apis
from shared.embeddings import get_text_embedding
from shared.database import search_by_text

def category_based_filter(query: str, results: list) -> list:
    """Simple category-based filtering as fallback when LLM fails"""
    try:
        # Determine expected category from query
        expected_category = None
        query_lower = query.lower()

        if "خاتم" in query_lower:
            expected_category = "خواتم"
        elif "عقد" in query_lower or "سلسلة" in query_lower or "سلسال" in query_lower:
            expected_category = "عقود"
        elif "أقراط" in query_lower or "قرط" in query_lower:
            expected_category = "أقراط"
        elif "سوار" in query_lower or "أساور" in query_lower:
            expected_category = "أساور"
        elif "دبوس" in query_lower or "دبابيس" in query_lower:
            expected_category = "دبابيس"
        elif "طقم" in query_lower or "أطقم" in query_lower:
            expected_category = "طقم"

        # If specific category detected, filter by it
        if expected_category:
            filtered = [r for r in results if r.metadata.get('category') == expected_category]
            if filtered:
                return filtered[:5]  # Top 5 in correct category

        # Otherwise return top results by similarity
        return results[:5]

    except Exception as e:
        print(f"Category filtering error: {e}")
        return results[:5]

def test_fallback_system():
    print("🔧 Testing Category-Based Fallback System")
    print("=" * 50)

    try:
        openai_client, pinecone_index = init_apis()

        # Test queries that should show category filtering working
        test_queries = [
            "خاتم",      # Should only return rings
            "خاتم هندسي", # Should only return rings
            "عقد",       # Should only return necklaces
            "عقد ذهبي"    # Should only return necklaces
        ]

        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            print("-" * 30)

            # Get search results
            results = search_by_text(pinecone_index, query, top_k=10, min_score=0.3)

            if not results:
                print("❌ No results found")
                continue

            print(f"📊 Found {len(results)} raw results")

            # Apply category-based filtering
            filtered_results = category_based_filter(query, results)

            print(f"✅ Category filter returned {len(filtered_results)} results")

            # Check if results match expected category
            expected_category = None
            if "خاتم" in query.lower():
                expected_category = "خواتم"
            elif "عقد" in query.lower():
                expected_category = "عقود"

            for result in filtered_results:
                name = result.metadata.get('name', 'N/A')
                category = result.metadata.get('category', 'N/A')
                score = result.score

                # Check if category matches expectation
                type_match = "✅" if (not expected_category or category == expected_category) else "❌"

                print(f"   {type_match} {name} ({category}) - Score: {score:.3f}")

                if expected_category and category != expected_category:
                    print(f"      🚨 TYPE MISMATCH: Expected {expected_category}, got {category}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_fallback_system()