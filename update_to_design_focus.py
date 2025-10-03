#!/usr/bin/env python3
"""
Update existing products to use design-focused descriptions
"""

import sys
sys.path.append('/home/hussein/shop-assistant')

from shared.config import init_apis
from shared.embeddings import get_text_embedding
import json

def extract_design_features(openai_client, product_name, current_description):
    """Extract design features from product name and description"""
    try:
        design_prompt = f"""
من اسم المنتج والوصف التالي، استخرج فقط ميزات التصميم والشكل البصري:

اسم المنتج: {product_name}
الوصف الحالي: {current_description}

اكتب وصفاً يركز فقط على:
- الشكل العام (دائري، مربع، مستطيل، قلب، نجمة، هندسي)
- النمط (كلاسيكي، عصري، فينتاج، بسيط، معقد)
- التفاصيل البصرية (متماثل، منحنيات، خطوط، أنماط)
- التركيب (قطعة واحدة، متصل، سلاسل)

تجاهل تماماً: المواد، الألوان، الأوزان، الأسعار، العيارات

أمثلة:
- "تصميم على شكل قلب بتفاصيل منحنية ناعمة"
- "شكل مستطيل هندسي مفتوح في الوسط"
- "تصميم فراشة متماثل مع أجنحة مقوسة"

أجب بجملة واحدة مختصرة تصف التصميم فقط.
أجب باللغة العربية فقط.
"""

        response = openai_client.chat.completions.create(
            model="gpt-5-nano-2025-08-07",
            messages=[{"role": "user", "content": design_prompt}]
        )

        design_description = response.choices[0].message.content.strip()
        return design_description

    except Exception as e:
        print(f"Error extracting design: {e}")
        # Fallback based on name
        if "قلب" in product_name:
            return "تصميم على شكل قلب"
        elif "فراشة" in product_name:
            return "تصميم على شكل فراشة"
        elif "نجمة" in product_name:
            return "تصميم على شكل نجمة"
        elif "دائري" in product_name or "حلقة" in product_name:
            return "تصميم دائري بسيط"
        elif "مستطيل" in product_name:
            return "تصميم مستطيل هندسي"
        else:
            return f"تصميم {product_name.split()[0] if product_name else 'بسيط'}"

def update_to_design_focus():
    print("🎨 Updating Products to Design-Focused Descriptions")
    print("=" * 60)

    try:
        openai_client, pinecone_index = init_apis()

        # Get all products
        dummy_embedding = get_text_embedding("مجوهرات")
        if not dummy_embedding:
            print("❌ Failed to get embedding")
            return

        all_results = pinecone_index.query(
            vector=dummy_embedding,
            top_k=100,
            include_metadata=True
        )

        print(f"📊 Found {len(all_results.matches)} products to update")

        updated_count = 0

        for result in all_results.matches:
            name = result.metadata.get('name', '')
            current_desc = result.metadata.get('description', '')

            print(f"\n🔧 Processing: {name}")

            # Extract design-focused description
            design_description = extract_design_features(openai_client, name, current_desc)

            # Add category context (keeping this minimal)
            category = result.metadata.get('category', '')
            if category:
                final_description = f"{category}: {design_description}"
            else:
                final_description = design_description

            print(f"   Design focus: {final_description}")

            # Generate new embedding
            new_embedding = get_text_embedding(final_description)

            if new_embedding:
                # Update metadata
                updated_metadata = result.metadata.copy()
                updated_metadata['description'] = final_description

                # Update in Pinecone
                pinecone_index.upsert(vectors=[{
                    "id": result.id,
                    "values": new_embedding,
                    "metadata": updated_metadata
                }])

                updated_count += 1
                print(f"   ✅ Updated")
            else:
                print(f"   ❌ Failed to generate embedding")

        print(f"\n🎉 Update completed! Updated {updated_count} products")
        print("\n📋 All products now focus on design/style matching rather than material matching")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Ask for confirmation
    print("⚠️  This will update ALL products in the database to focus on design features.")
    print("   Materials and colors will be de-emphasized in favor of visual design matching.")
    print()
    confirm = input("Continue? (y/N): ")

    if confirm.lower() == 'y':
        update_to_design_focus()
    else:
        print("❌ Operation cancelled")