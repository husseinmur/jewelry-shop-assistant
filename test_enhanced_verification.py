#!/usr/bin/env python3
"""
Test the enhanced LLM verification system
"""

import sys
import os
sys.path.append('/home/hussein/shop-assistant')

import time
from shared.config import init_apis
from shared.embeddings import get_text_embedding
import openai
import json

def llm_filter_results_enhanced(query: str, results: list, openai_client) -> list:
    """Enhanced LLM filtering with better context"""
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

Task: Return product IDs that would satisfy the customer's search intent.

IMPORTANT CONTEXT & FLEXIBILITY RULES:
📿 JEWELRY USE CASES - Understand customer intent:
- "خاتم خطوبة" (engagement ring) = elegant rings with stones, gold, suitable for proposals
- "خاتم زواج" (wedding ring) = classic rings, bands, gold/platinum, suitable for marriage
- "خاتم بسيط" (simple ring) = minimal design, clean lines, not overly decorative
- "عقد فاخر" (luxury necklace) = high-end necklaces, precious materials, sophisticated design
- "أقراط يومية" (daily earrings) = comfortable, suitable for everyday wear
- "مجوهرات هدية" (gift jewelry) = presentable pieces, nice packaging appeal

🎯 MATCHING STRATEGY:
- Focus on SUITABILITY for the intended use, not exact terminology
- A beautiful gold ring with stones IS suitable for engagement even if not labeled "engagement ring"
- A simple gold band IS suitable for wedding even if not labeled "wedding ring"
- Consider material, design style, and appropriateness for the occasion

💎 MATERIAL & STYLE UNDERSTANDING:
- "ذهب" includes all gold types (yellow, white, rose gold)
- "بسيط" means clean, minimal, not overly decorative
- "فاخر" means luxury materials, sophisticated design, higher quality
- "أنيق" means elegant, refined, sophisticated

✅ EXAMPLES:
Query: "خاتم خطوبة" → Return: elegant rings with stones, gold rings suitable for proposals
Query: "سلسلة بسيطة" → Return: minimal necklaces, clean design chains
Query: "أقراط ذهب" → Return: any gold earrings regardless of specific style

❌ ONLY EXCLUDE if products are completely wrong category or material
- Query: "خاتم" (ring) → Don't return necklaces or earrings
- Query: "ذهب" (gold) → Don't return silver-only items

Return ONLY a JSON list of product IDs that match the customer's intent: ["id1", "id2", "id3"] or []
"""

        response = openai_client.chat.completions.create(
            model="gpt-5-nano-2025-08-07",
            messages=[{"role": "user", "content": verification_prompt}],
            temperature=1.0,
            max_completion_tokens=2000
        )

        # Parse response to get filtered IDs
        response_text = response.choices[0].message.content.strip()
        try:
            filtered_ids = json.loads(response_text)
            if not isinstance(filtered_ids, list):
                filtered_ids = []
        except Exception as e:
            print(f"    JSON parsing error: {e}")
            print(f"    Raw response: {response_text}")
            filtered_ids = []

        # Filter original results by verified IDs
        filtered_results = [r for r in results if r.id in filtered_ids]
        return filtered_results

    except Exception as e:
        print(f"Error in LLM filtering: {e}")
        return []

def test_enhanced_verification():
    print("🧪 Testing Enhanced LLM Verification")
    print("=" * 50)

    try:
        # Initialize
        openai_client, pinecone_index = init_apis()
        print("✅ APIs initialized successfully")

        # Test problematic queries
        test_queries = [
            "خاتم خطوبة",  # The problematic one
            "خاتم زواج",
            "خاتم بسيط",
            "خاتم ذهب",
            "سلسلة بسيطة"
        ]

        for query in test_queries:
            print(f"\n🔍 Testing: '{query}'")
            print("-" * 30)

            start_time = time.time()

            # Get embedding and search
            query_embedding = get_text_embedding(query)
            if not query_embedding:
                print("❌ Failed to get embedding")
                continue

            # Search Pinecone
            pinecone_results = pinecone_index.query(
                vector=query_embedding,
                top_k=5,
                include_metadata=True
            )

            print(f"Pinecone found: {len(pinecone_results.matches)} results")

            # Apply enhanced LLM filter
            filtered_results = llm_filter_results_enhanced(query, pinecone_results.matches, openai_client)

            total_time = time.time() - start_time

            print(f"LLM filter kept: {len(filtered_results)} results")
            print(f"Total time: {total_time:.2f}s")

            if filtered_results:
                print("✅ Results:")
                for i, result in enumerate(filtered_results[:3], 1):
                    name = result.metadata.get('name', 'N/A')
                    score = result.score
                    print(f"  {i}. {name} (score: {score:.3f})")
            else:
                print("❌ No results passed filter")

            print()

        print("🎉 Enhanced verification testing completed!")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_enhanced_verification()