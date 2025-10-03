#!/usr/bin/env python3
"""
Test the type verification system to ensure proper category matching
"""

import sys
sys.path.append('/home/hussein/shop-assistant')

from shared.config import init_apis
from shared.embeddings import get_text_embedding, expand_search_query, parse_query_expansion
from shared.database import search_by_text
import openai

def test_type_verification():
    print("🧪 Testing Type Verification System")
    print("=" * 50)

    try:
        openai_client, pinecone_index = init_apis()

        # Test queries that should show type verification working
        test_queries = [
            "خاتم هندسي",  # Should only return rings
            "عقد هندسي",   # Should only return necklaces
            "خاتم بشكل قلب", # Should only return rings
            "عقد بتصميم دائري" # Should only return necklaces
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

            # Test LLM verification (simulating the chatbot's verification)
            verification_prompt = f"""
Task: Return product IDs that match the customer's search intent with STRICT TYPE CHECKING.

🚨 CRITICAL TYPE MATCHING RULES:
1️⃣ **JEWELRY TYPE MUST MATCH EXACTLY:**
- خاتم/ring queries → ONLY return rings (خواتم category)
- عقد/necklace queries → ONLY return necklaces (عقود category)
- أقراط/earrings queries → ONLY return earrings (أقراط category)
- أساور/bracelet queries → ONLY return bracelets (أساور category)

❌ **NEVER mix types**: If customer asks for rings, DO NOT return necklaces even if design matches
✅ **Design matching within type**: Match design features ONLY within the correct jewelry type

Customer Query: "{query}"

Products to evaluate:
"""

            for i, result in enumerate(results, 1):
                name = result.metadata.get('name', 'N/A')
                category = result.metadata.get('category', 'N/A')
                description = result.metadata.get('description', 'N/A')
                score = result.score

                verification_prompt += f"""
{i}. ID: {result.id}
   Name: {name}
   Category: {category}
   Description: {description}
   Similarity: {score:.3f}
"""

            verification_prompt += """

Return ONLY the product IDs that match both the design intent AND the correct jewelry type.
Format: Just list the IDs, one per line. If no products match the type requirement, return "NO_MATCHES".
"""

            # Get LLM verification
            response = openai_client.chat.completions.create(
                model="gpt-5-nano-2025-08-07",
                messages=[{"role": "user", "content": verification_prompt}],
                max_completion_tokens=500
            )

            verified_ids = response.choices[0].message.content.strip().split('\n')
            verified_ids = [id.strip() for id in verified_ids if id.strip() and id.strip() != "NO_MATCHES"]

            print(f"✅ LLM verified {len(verified_ids)} products")

            # Show verification results
            for result in results:
                if result.id in verified_ids:
                    name = result.metadata.get('name', 'N/A')
                    category = result.metadata.get('category', 'N/A')
                    description = result.metadata.get('description', 'N/A')[:100]
                    score = result.score

                    # Check if type matches query expectation
                    expected_type = ""
                    if "خاتم" in query:
                        expected_type = "خواتم"
                    elif "عقد" in query:
                        expected_type = "عقود"
                    elif "أقراط" in query:
                        expected_type = "أقراط"
                    elif "أساور" in query:
                        expected_type = "أساور"

                    type_match = "✅" if category == expected_type else "❌"

                    print(f"   {type_match} {name} ({category}) - Score: {score:.3f}")
                    print(f"      {description}...")

                    if category != expected_type:
                        print(f"      🚨 TYPE MISMATCH: Expected {expected_type}, got {category}")

            print()

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_type_verification()