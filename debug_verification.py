#!/usr/bin/env python3
"""
Debug the type verification system
"""

import sys
sys.path.append('/home/hussein/shop-assistant')

from shared.config import init_apis
from shared.embeddings import get_text_embedding
from shared.database import search_by_text
import openai

def debug_verification():
    print("🐛 Debugging Type Verification")
    print("=" * 40)

    try:
        openai_client, pinecone_index = init_apis()

        # First, let's see what's actually in the database
        print("📊 Database Contents:")
        print("-" * 20)

        dummy_embedding = get_text_embedding("مجوهرات")
        all_results = pinecone_index.query(
            vector=dummy_embedding,
            top_k=20,
            include_metadata=True
        )

        categories = {}
        for result in all_results.matches:
            name = result.metadata.get('name', 'N/A')
            category = result.metadata.get('category', 'N/A')
            description = result.metadata.get('description', 'N/A')[:80]

            if category not in categories:
                categories[category] = []
            categories[category].append(f"   • {name} - {description}...")

        for category, items in categories.items():
            print(f"\n{category}:")
            for item in items[:3]:  # Show first 3 items per category
                print(item)
            if len(items) > 3:
                print(f"   ... and {len(items)-3} more")

        # Now test a simple ring query
        print(f"\n🔍 Testing simple ring query: 'خاتم'")
        print("-" * 30)

        results = search_by_text(pinecone_index, "خاتم", top_k=8, min_score=0.3)

        if not results:
            print("❌ No results found for 'خاتم'")
            return

        print(f"📊 Found {len(results)} results for 'خاتم':")
        for i, result in enumerate(results, 1):
            name = result.metadata.get('name', 'N/A')
            category = result.metadata.get('category', 'N/A')
            description = result.metadata.get('description', 'N/A')[:100]
            score = result.score
            print(f"{i}. {name} ({category}) - Score: {score:.3f}")
            print(f"   Description: {description}...")
            print()

        # Test LLM verification with detailed debugging
        print("🤖 Testing LLM Verification:")
        print("-" * 30)

        results_text = ""
        for i, result in enumerate(results, 1):
            name = result.metadata.get('name', 'N/A')
            category = result.metadata.get('category', 'N/A')
            description = result.metadata.get('description', 'N/A')
            score = result.score

            results_text += f"ID: {result.id}\n"
            results_text += f"Name: {name}\n"
            results_text += f"Category: {category}\n"
            results_text += f"Description: {description[:100]}...\n"
            results_text += f"Score: {score:.3f}\n\n"

        verification_prompt = f"""
Query: "خاتم"

Products:
{results_text}

Rules:
- خاتم queries → only خواتم category
- عقد queries → only عقود category
- أقراط queries → only أقراط category
- أساور queries → only أساور category

Return JSON list of matching product IDs: ["id1", "id2"] or []
"""

        print("Verification prompt:")
        print(verification_prompt[:500] + "...")
        print()

        response = openai_client.chat.completions.create(
            model="gpt-5-nano-2025-08-07",
            messages=[{"role": "user", "content": verification_prompt}],
            max_completion_tokens=500
        )

        verification_response = response.choices[0].message.content.strip()
        print(f"LLM Response: '{verification_response}'")

        if verification_response == "NO_MATCHES":
            print("❌ LLM returned NO_MATCHES")
            print("🔍 Let's check if there are any خواتم in the results:")
            for result in results:
                category = result.metadata.get('category', 'N/A')
                name = result.metadata.get('name', 'N/A')
                if category == "خواتم":
                    print(f"   ✅ Found ring: {name} ({category})")
        else:
            verified_ids = verification_response.split('\n')
            verified_ids = [id.strip() for id in verified_ids if id.strip()]
            print(f"✅ LLM verified {len(verified_ids)} products: {verified_ids}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    debug_verification()