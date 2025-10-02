#!/usr/bin/env python3
"""
Test script for the optimized search system
"""

import time
import sys
import os
sys.path.append('/home/hussein/shop-assistant')

from shared.config import init_apis
from shared.embeddings import get_text_embedding
import openai
import json

def llm_filter_results(query: str, results: list, openai_client) -> list:
    """Use LLM to intelligently filter search results for relevance"""
    try:
        if not results:
            return []

        # Format results for verification
        results_text = ""
        for i, result in enumerate(results):
            metadata = result.metadata
            results_text += f"ID: {result.id}\n"
            results_text += f"Name: {metadata.get('name', 'N/A')}\n"
            results_text += f"Category: {metadata.get('category', 'N/A')}\n"
            results_text += f"Description: {metadata.get('description', 'N/A')[:100]}...\n"
            results_text += f"Score: {result.score:.3f}\n\n"

        verification_prompt = f"""
Query: "{query}"

Available products to evaluate:
{results_text}

Task: Return only the product IDs that truly match the search query.

Rules:
- If query is "خاتم ذهب" (gold ring), only return actual gold rings
- If query is "سلسلة بسيطة" (simple necklace), only return simple/minimalist necklaces
- If query is "مجوهرات غالية" (expensive jewelry), return high-priced items
- Consider both name, category, and description for matching
- If NO products truly match the query intent, return empty list
- Only return products that a customer would actually want for this search

Return ONLY a JSON list of product IDs, nothing else: ["id1", "id2", "id3"] or []
"""

        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": verification_prompt}],
            temperature=0.1,
            max_tokens=500  # Increased for longer UUID lists
        )

        # Parse response to get filtered IDs
        response_text = response.choices[0].message.content.strip()
        print(f"    LLM raw response: '{response_text[:100]}...'")
        try:
            filtered_ids = json.loads(response_text)
            if not isinstance(filtered_ids, list):
                print(f"    Warning: LLM response is not a list: {type(filtered_ids)}")
                filtered_ids = []
        except Exception as e:
            print(f"    JSON parsing error: {e}")
            filtered_ids = []

        # Filter original results by verified IDs
        filtered_results = [r for r in results if r.id in filtered_ids]
        print(f"    LLM returned {len(filtered_ids)} IDs, matched {len(filtered_results)} results")
        return filtered_results

    except Exception as e:
        print(f"Error in LLM filtering: {e}")
        # Fallback to similarity filtering
        fallback_results = [r for r in results if r.score >= 0.4][:5]
        print(f"    Using fallback filter, kept {len(fallback_results)} results")
        return fallback_results

def test_optimized_search(query: str, openai_client, pinecone_index):
    """Test the optimized search with performance timing"""

    print(f"\n🔍 Testing query: '{query}'")
    start_time = time.time()

    try:
        # Step 1: Get embedding
        embedding_start = time.time()
        query_embedding = get_text_embedding(query)
        embedding_time = time.time() - embedding_start

        if not query_embedding:
            print("❌ Failed to get embedding")
            return None

        print(f"   ⚡ Embedding: {embedding_time:.2f}s")

        # Step 2: Pinecone search
        pinecone_start = time.time()
        pinecone_results = pinecone_index.query(
            vector=query_embedding,
            top_k=15,
            include_metadata=True
        )
        pinecone_time = time.time() - pinecone_start
        print(f"   ⚡ Pinecone: {pinecone_time:.2f}s (found {len(pinecone_results.matches)})")

        # Step 3: Basic similarity filter
        filter_start = time.time()
        decent_results = [r for r in pinecone_results.matches if r.score >= 0.3]
        filter_time = time.time() - filter_start
        print(f"   ⚡ Basic filter: {filter_time:.3f}s (kept {len(decent_results)})")

        if not decent_results:
            total_time = time.time() - start_time
            print(f"   ❌ No decent matches found")
            print(f"   🕒 Total time: {total_time:.2f}s")
            return None

        # Step 4: LLM verification
        llm_start = time.time()
        filtered_results = llm_filter_results(query, decent_results, openai_client)
        llm_time = time.time() - llm_start
        print(f"   ⚡ LLM filter: {llm_time:.2f}s (kept {len(filtered_results)})")

        total_time = time.time() - start_time
        print(f"   🕒 Total time: {total_time:.2f}s")

        if filtered_results:
            print(f"   ✅ Results:")
            for i, result in enumerate(filtered_results[:3], 1):
                name = result.metadata.get('name', 'N/A')
                price = result.metadata.get('price', 0)
                score = result.score
                print(f"      {i}. {name} - {price} ريال (score: {score:.3f})")
        else:
            print(f"   ❌ No results passed LLM filter")

        return {
            'query': query,
            'total_time': total_time,
            'embedding_time': embedding_time,
            'pinecone_time': pinecone_time,
            'filter_time': filter_time,
            'llm_time': llm_time,
            'results_count': len(filtered_results),
            'results': filtered_results
        }

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

def main():
    print("🧪 Testing Optimized Search System")
    print("=" * 50)

    try:
        # Initialize
        print("🔧 Initializing APIs...")
        openai_client, pinecone_index = init_apis()
        print("✅ APIs initialized successfully")

        # Test queries
        test_queries = [
            "خاتم ذهب",
            "سلسلة بسيطة",
            "أقراط فضة",
            "مجوهرات غالية",
            "خاتم زواج"
        ]

        results = []

        for query in test_queries:
            result = test_optimized_search(query, openai_client, pinecone_index)
            if result:
                results.append(result)

        # Performance summary
        print("\n📊 Performance Summary")
        print("=" * 50)

        if results:
            avg_total = sum(r['total_time'] for r in results) / len(results)
            avg_embedding = sum(r['embedding_time'] for r in results) / len(results)
            avg_pinecone = sum(r['pinecone_time'] for r in results) / len(results)
            avg_llm = sum(r['llm_time'] for r in results) / len(results)

            print(f"Average total time: {avg_total:.2f}s")
            print(f"  - Embedding: {avg_embedding:.2f}s ({avg_embedding/avg_total*100:.1f}%)")
            print(f"  - Pinecone: {avg_pinecone:.2f}s ({avg_pinecone/avg_total*100:.1f}%)")
            print(f"  - LLM filter: {avg_llm:.2f}s ({avg_llm/avg_total*100:.1f}%)")

            successful_queries = sum(1 for r in results if r['results_count'] > 0)
            print(f"\nSuccess rate: {successful_queries}/{len(results)} ({successful_queries/len(results)*100:.1f}%)")

        print("\n🎉 Testing completed!")

    except Exception as e:
        print(f"❌ Failed to initialize: {e}")

if __name__ == "__main__":
    main()