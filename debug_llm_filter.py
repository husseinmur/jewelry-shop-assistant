#!/usr/bin/env python3
"""
Debug the LLM filtering to see why it's rejecting all results
"""

import sys
import os
sys.path.append('/home/hussein/shop-assistant')

from shared.config import init_apis
from shared.embeddings import get_text_embedding
import openai
import json

def debug_llm_filter():
    print("🐛 Debugging LLM Filter")
    print("=" * 50)

    try:
        # Initialize
        openai_client, pinecone_index = init_apis()

        # Test with a simple query
        query = "خاتم ذهب"
        print(f"Testing query: '{query}'")

        # Get embedding and search
        query_embedding = get_text_embedding(query)
        pinecone_results = pinecone_index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )

        print(f"\nFound {len(pinecone_results.matches)} results:")
        results_text = ""
        for i, result in enumerate(pinecone_results.matches):
            metadata = result.metadata
            print(f"  {i+1}. {metadata.get('name', 'N/A')} (score: {result.score:.3f})")
            print(f"      Category: {metadata.get('category', 'N/A')}")
            print(f"      Description: {metadata.get('description', 'N/A')[:100]}...")

            results_text += f"ID: {result.id}\n"
            results_text += f"Name: {metadata.get('name', 'N/A')}\n"
            results_text += f"Category: {metadata.get('category', 'N/A')}\n"
            results_text += f"Description: {metadata.get('description', 'N/A')[:100]}...\n"
            results_text += f"Score: {result.score:.3f}\n\n"

        # Create the LLM prompt
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

        print(f"\n📝 LLM Prompt:")
        print("-" * 30)
        print(verification_prompt)
        print("-" * 30)

        # Send to LLM
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": verification_prompt}],
            temperature=0.1,
            max_tokens=200
        )

        llm_response = response.choices[0].message.content.strip()
        print(f"\n🤖 LLM Response:")
        print(f"'{llm_response}'")

        # Try to parse
        try:
            filtered_ids = json.loads(llm_response)
            print(f"\n✅ Parsed as JSON: {filtered_ids}")
            print(f"Type: {type(filtered_ids)}")
            print(f"Length: {len(filtered_ids) if isinstance(filtered_ids, list) else 'N/A'}")
        except Exception as e:
            print(f"\n❌ JSON parsing failed: {e}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_llm_filter()