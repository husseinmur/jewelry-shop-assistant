#!/usr/bin/env python3
"""
Fix corrupted product descriptions in Pinecone database
"""

import sys
sys.path.append('/home/hussein/shop-assistant')

from shared.config import init_apis
from shared.embeddings import get_text_embedding
import json

def fix_corrupted_descriptions():
    print("🔧 Fixing Corrupted Product Descriptions")
    print("=" * 50)

    try:
        openai_client, pinecone_index = init_apis()

        # Search for all products to find corrupted ones
        print("🔍 Scanning database for corrupted entries...")

        # Get a broad search to find all products
        dummy_embedding = get_text_embedding("مجوهرات")
        if not dummy_embedding:
            print("❌ Failed to get embedding")
            return

        all_results = pinecone_index.query(
            vector=dummy_embedding,
            top_k=100,  # Get many results
            include_metadata=True
        )

        corrupted_items = []

        for result in all_results.matches:
            description = result.metadata.get('description', '')
            name = result.metadata.get('name', 'N/A')

            # Check if description contains prompt text (corrupted)
            if ('أوصِف هذه القطعة من المجوهرات' in description or
                'كما لو كنت تكتب وصفاً في كتالوج' in description):
                corrupted_items.append({
                    'id': result.id,
                    'name': name,
                    'description': description[:200] + '...',
                    'metadata': result.metadata
                })
                print(f"🔍 Found corrupted: {name} (ID: {result.id})")

        print(f"\n📊 Found {len(corrupted_items)} corrupted items")

        if not corrupted_items:
            print("✅ No corrupted items found!")
            return

        # Fix each corrupted item
        for item in corrupted_items:
            print(f"\n🔧 Fixing: {item['name']}")

            # Create a proper description based on the name and available metadata
            fixed_description = create_proper_description(item['metadata'], openai_client)

            if fixed_description:
                # Generate new embedding
                new_embedding = get_text_embedding(fixed_description)

                if new_embedding:
                    # Update metadata
                    updated_metadata = item['metadata'].copy()
                    updated_metadata['description'] = fixed_description

                    # Update in Pinecone
                    pinecone_index.upsert(vectors=[{
                        "id": item['id'],
                        "values": new_embedding,
                        "metadata": updated_metadata
                    }])

                    print(f"✅ Fixed: {item['name']}")
                    print(f"   New description: {fixed_description[:100]}...")
                else:
                    print(f"❌ Failed to generate embedding for: {item['name']}")
            else:
                print(f"❌ Failed to generate description for: {item['name']}")

        print(f"\n🎉 Cleanup completed! Fixed {len(corrupted_items)} items")

    except Exception as e:
        print(f"❌ Error: {e}")

def create_proper_description(metadata, openai_client):
    """Create a proper description from metadata"""
    try:
        name = metadata.get('name', '')
        category = metadata.get('category', '')
        karat = metadata.get('karat', '')
        weight = metadata.get('weight', 0)
        design = metadata.get('design', '')
        style = metadata.get('style', '')

        # Build description from available metadata
        description_parts = []

        # Start with name
        if name:
            description_parts.append(name)

        # Add category info
        if category:
            description_parts.append(f"من فئة {category}")

        # Add material info
        if karat:
            description_parts.append(f"مصنوع من {karat}")

        # Add weight
        if weight and weight > 0:
            description_parts.append(f"وزن {weight} جرام")

        # Add design
        if design:
            description_parts.append(f"بتصميم {design}")

        # Add style
        if style:
            description_parts.append(f"بطراز {style}")

        # Create initial description
        basic_description = " ".join(description_parts)

        # If we have enough info, return it
        if len(description_parts) >= 3:
            return basic_description

        # Otherwise, try to enhance with AI (as fallback)
        enhancement_prompt = f"""
أكتب وصفاً مختصراً وواقعياً لقطعة المجوهرات التالية:

الاسم: {name}
الفئة: {category}
المعلومات المتوفرة: {basic_description}

أكتب وصفاً بسيطاً ومناسباً لكتالوج متجر مجوهرات. استخدم 2-3 جمل فقط.
أجب باللغة العربية فقط.
"""

        response = openai_client.chat.completions.create(
            model="gpt-5-nano-2025-08-07",
            messages=[{"role": "user", "content": enhancement_prompt}]
        )

        enhanced_description = response.choices[0].message.content.strip()
        return enhanced_description

    except Exception as e:
        print(f"Error creating description: {e}")
        # Fallback to basic description
        return f"{metadata.get('name', 'قطعة مجوهرات')} من {metadata.get('category', 'المجوهرات')}"

if __name__ == "__main__":
    fix_corrupted_descriptions()