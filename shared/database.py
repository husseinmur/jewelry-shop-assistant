import streamlit as st
import uuid
from .embeddings import get_image_description, get_text_embedding, get_image_category

def store_product(index, image, name, price, category, image_url=None, additional_info="", karat="", weight=0.0, design="", style="", product_url=""):
    """Store a product in Pinecone with embeddings"""
    try:
        # Generate unique ID
        product_id = str(uuid.uuid4())
        
        # Get description from image
        ai_description = get_image_description(image)
        
        # Combine AI description with additional info
        if additional_info.strip():
            description = f"{ai_description}\n\nتفاصيل إضافية: {additional_info.strip()}"
        else:
            description = ai_description
        
        # Add jewelry details to description for better search
        jewelry_details = []
        if karat: jewelry_details.append(f"العيار: {karat}")
        if weight > 0: jewelry_details.append(f"الوزن: {weight} جرام")
        if design: jewelry_details.append(f"التصميم: {design}")
        if style: jewelry_details.append(f"الستايل: {style}")
        
        if jewelry_details:
            description += f"\n\nمواصفات: {' | '.join(jewelry_details)}"
        
        # Get embedding from description
        embedding = get_text_embedding(description)
        
        if embedding is None:
            st.error("فشل في توليد التضمين")
            return False
        
        # Prepare metadata
        metadata = {
            "name": name,
            "price": float(price),
            "category": category,
            "description": description,
            "image_url": image_url or f"product_{product_id}.jpg",
            "karat": karat,
            "weight": float(weight) if weight > 0 else 0.0,
            "design": design,
            "style": style,
            "product_url": product_url
        }
        
        # Store in Pinecone
        index.upsert(vectors=[{
            "id": product_id,
            "values": embedding,
            "metadata": metadata
        }])
        
        st.success(f"✅ تم حفظ: {name}")
        return True
        
    except Exception as e:
        st.error(f"خطأ في حفظ المنتج: {e}")
        return False

def search_products(index, query_embedding, top_k=10, min_score=0.3):
    """Search for similar products using embedding"""
    try:
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        # Filter by minimum similarity score
        filtered_results = [
            result for result in results.matches 
            if result.score >= min_score
        ]
        
        return filtered_results
        
    except Exception as e:
        st.error(f"خطأ في البحث عن المنتجات: {e}")
        return []

def search_by_text(index, text_query, top_k=10, min_score=0.3):
    """Search products by text query"""
    try:
        # Get embedding for the text query
        embedding = get_text_embedding(text_query)
        
        if embedding is None:
            return []
        
        return search_products(index, embedding, top_k, min_score)
        
    except Exception as e:
        st.error(f"خطأ في البحث النصي: {e}")
        return []

def search_by_image(index, image, top_k=10, min_score=0.5):
    """Search products by uploaded image using description + category filtering"""
    try:
        # Method 1: Get category from image
        detected_category = get_image_category(image)
        st.info(f"🎯 تم تحديد الفئة من الصورة: {detected_category}")
        
        # Method 2: Get description and search by text
        description = get_image_description(image)
        
        # Get more results to filter by category
        text_results = search_by_text(index, description, top_k * 3, min_score=0.3)
        
        # Filter by detected category first
        if detected_category != "أخرى":
            category_filtered = [
                result for result in text_results 
                if result.metadata.get('category', '').lower() == detected_category.lower()
            ]
            
            if len(category_filtered) >= 1:
                st.success(f"✅ تم العثور على {len(category_filtered)} عنصر في فئة {detected_category}")
                return category_filtered[:top_k]
            else:
                # No category matches - don't mix with other categories
                st.warning(f"⚠️ لم يتم العثور على منتجات في فئة {detected_category}. جرب وصف أكثر تفصيلاً أو أضف منتجات لهذه الفئة.")
                return []
        else:
            # No specific category detected, return general results
            return text_results[:top_k]
        
    except Exception as e:
        st.error(f"خطأ في البحث بالصورة: {e}")
        return []

def smart_search(index, query, search_type="text", top_k=10):
    """Semantic search with optional category preference"""
    try:
        if search_type == "text":
            # Detect category from query (but don't enforce it)
            jewelry_categories = {
                "خاتم": "خواتم", "خواتم": "خواتم",
                "قلادة": "عقود", "عقد": "عقود", "عقود": "عقود",
                "سلسلة": "عقود", "سلاسل": "عقود", "سلسال": "عقود",
                "أقراط": "أقراط", "قرط": "أقراط",
                "أساور": "أساور", "سوار": "أساور", "اسورة": "أساور",
                "دبابيس": "دبابيس", "دبوس": "دبابيس",
                "طقم": "طقم", "أطقم": "طقم"
            }

            detected_category = None
            query_lower = query.lower()
            for keyword, category in jewelry_categories.items():
                if keyword in query_lower:
                    detected_category = category
                    break

            # PRIMARY: Semantic search with stricter thresholds
            # Start with good threshold for quality matches
            all_results = search_by_text(index, query, top_k * 3, min_score=0.5)

            if not all_results:
                # If no results, try moderate threshold
                all_results = search_by_text(index, query, top_k * 2, min_score=0.35)

            if not all_results:
                return []

            # SECONDARY: Prefer category matches if detected, but don't exclude others
            if detected_category:
                # Split results into category matches and others
                category_matches = [
                    result for result in all_results
                    if result.metadata.get('category', '').lower() == detected_category.lower()
                ]
                other_matches = [
                    result for result in all_results
                    if result.metadata.get('category', '').lower() != detected_category.lower()
                ]

                if category_matches:
                    # Prioritize category matches but include others if needed
                    results = category_matches[:max(top_k//2, 3)]
                    remaining_slots = top_k - len(results)
                    if remaining_slots > 0:
                        results.extend(other_matches[:remaining_slots])
                    st.info(f"🎯 العثور على {len(category_matches)} منتج في فئة {detected_category}")
                else:
                    # No category matches - show all semantic matches
                    results = all_results[:top_k]
                    st.info(f"💡 لم يتم العثور على منتجات في فئة {detected_category}، عرض النتائج المشابهة")
            else:
                # No specific category - show best semantic matches
                results = all_results[:top_k]

            # Quality threshold: Only show good matches
            final_results = [r for r in results if r.score >= 0.4]

            if final_results:
                best_score = max(r.score for r in final_results)
                if best_score < 0.5:
                    st.warning("💡 النتائج قد تكون عامة. جرب وصف أكثر تفصيلاً لنتائج أفضل.")

            return final_results

        elif search_type == "image":
            # For image search, use the category-aware image search
            results = search_by_image(index, query, top_k, min_score=0.3)
            return results

        return []

    except Exception as e:
        st.error(f"خطأ في البحث الذكي: {e}")
        return []

def get_all_products(index, limit=100):
    """Get all products from the database (for admin view)"""
    try:
        # Note: This is a simple approach for demo
        # In production, you'd want pagination
        results = index.query(
            vector=[0.0] * 1536,  # dummy vector
            top_k=limit,
            include_metadata=True
        )
        
        return results.matches
        
    except Exception as e:
        st.error(f"خطأ في الحصول على المنتجات: {e}")
        return []

def delete_product(index, product_id):
    """Delete a product from the database"""
    try:
        index.delete(ids=[product_id])
        st.success("تم حذف المنتج")
        return True
        
    except Exception as e:
        st.error(f"خطأ في حذف المنتج: {e}")
        return False