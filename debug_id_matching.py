#!/usr/bin/env python3
"""
Debug ID matching issue
"""

import sys
import os
sys.path.append('/home/hussein/shop-assistant')

from shared.config import init_apis
from shared.embeddings import get_text_embedding

def debug_id_matching():
    print("🐛 Debugging ID Matching")
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

        print(f"\nPinecone results:")
        original_ids = []
        for i, result in enumerate(pinecone_results.matches):
            print(f"  {i+1}. ID: {result.id} (type: {type(result.id)})")
            print(f"      Name: {result.metadata.get('name', 'N/A')}")
            original_ids.append(result.id)

        # Simulate LLM response
        llm_ids = ['48291999-e4dc-4b98-946a-f23d5030507f', 'e46898a4-3214-4a24-8c0b-3919ef250485', 'd8890e87-0d39-4fe8-a804-7aa375a03341', 'e1a769ae-ee4d-44a7-bad8-8f78fe2317c5', '4ae6ceeb-7e2c-4150-bb8a-2cda74db98c6']

        print(f"\nLLM returned IDs:")
        for i, llm_id in enumerate(llm_ids):
            print(f"  {i+1}. {llm_id} (type: {type(llm_id)})")

        print(f"\nID Matching Test:")
        for original_id in original_ids:
            match_found = original_id in llm_ids
            print(f"  '{original_id}' in LLM list: {match_found}")
            if not match_found:
                # Check if there's a close match
                for llm_id in llm_ids:
                    if original_id == llm_id:
                        print(f"    Direct comparison with '{llm_id}': True")
                    else:
                        print(f"    Direct comparison with '{llm_id}': False")

        # Test the filtering logic
        filtered_results = [r for r in pinecone_results.matches if r.id in llm_ids]
        print(f"\nFiltered results count: {len(filtered_results)}")

        for i, result in enumerate(filtered_results):
            print(f"  {i+1}. {result.metadata.get('name', 'N/A')}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_id_matching()