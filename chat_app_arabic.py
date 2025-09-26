import streamlit as st
from PIL import Image
from shared.config import init_apis
from shared.database import search_by_text, search_by_image
from shared.embeddings import expand_search_query, parse_query_expansion

# Page config
st.set_page_config(
    page_title="مساعد متجر المجوهرات",
    page_icon="💎",
    layout="wide"
)

st.title("💎 مساعد متجر المجوهرات")
st.subheader("ابحث عن المجوهرات المثالية لك")

# Initialize APIs
try:
    openai_client, pinecone_index = init_apis()
except Exception as e:
    st.error(f"خطأ في الاتصال: {e}")
    st.stop()

# Main search interface
st.markdown("### كيف يمكنني مساعدتك اليوم؟")

# Two columns for different search methods
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🔍 البحث بالنص")
    search_query = st.text_input(
        "صف ما تبحث عنه:",
        placeholder="مثال: قلادة ذهبية بشكل الياسمين، خاتم فضي بسيط..."
    )
    
    if st.button("🔍 ابحث بالنص", type="primary"):
        if search_query:
            with st.spinner("البحث عن المنتجات..."):
                # Expand the query
                expansion = expand_search_query(search_query)
                parsed = parse_query_expansion(expansion)
                
                # Show what we're searching for
                st.info(f"🔍 البحث عن: {parsed['primary']}")
                if parsed['related']:
                    st.write(f"**مصطلحات ذات صلة:** {', '.join(parsed['related'])}")
                
                # Search using expanded query
                all_terms = [parsed['primary']] + parsed['related']
                search_text = ' '.join(all_terms)
                
                results = search_by_text(pinecone_index, search_text, top_k=8)
                
                if results:
                    st.success(f"✅ تم العثور على {len(results)} منتج")
                    
                    # Display results
                    cols = st.columns(2)
                    for idx, result in enumerate(results):
                        with cols[idx % 2]:
                            metadata = result.metadata
                            
                            # Product card
                            st.markdown("---")
                            st.subheader(metadata.get('name', 'منتج'))
                            st.write(f"**💰 السعر:** {metadata.get('price', 0):.2f} ريال")
                            st.write(f"**📝 الفئة:** {metadata.get('category', 'غير محدد')}")
                            
                            # Description
                            description = metadata.get('description', '')
                            if len(description) > 150:
                                description = description[:150] + "..."
                            st.write(f"**🔍 الوصف:** {description}")
                            
                            # Match score
                            match_percentage = result.score * 100
                            st.write(f"**🎯 درجة التطابق:** {match_percentage:.1f}%")
                            
                            # Add to cart button (placeholder)
                            if st.button(f"🛒 أضف للسلة", key=f"cart_{result.id}"):
                                st.success("تم إضافة المنتج للسلة! 🛍️")
                else:
                    st.warning("لم يتم العثور على منتجات مطابقة. جرب مصطلحات بحث مختلفة.")

with col2:
    st.markdown("#### 📸 البحث بالصورة")
    uploaded_image = st.file_uploader(
        "ارفع صورة منتج مشابه:",
        type=['png', 'jpg', 'jpeg'],
        help="ارفع صورة لقطعة مجوهرات تريد العثور على شيء مشابه لها"
    )
    
    if uploaded_image:
        # Display uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="الصورة المرفوعة", use_container_width=True)
        
        if st.button("🔍 ابحث بالصورة", type="primary"):
            with st.spinner("تحليل الصورة والبحث..."):
                # Search by image
                results = search_by_image(pinecone_index, image, top_k=6)
                
                if results:
                    st.success(f"✅ تم العثور على {len(results)} منتج مشابه")
                    
                    # Display results in grid
                    for idx, result in enumerate(results):
                        metadata = result.metadata
                        
                        # Product card
                        st.markdown("---")
                        st.subheader(metadata.get('name', 'منتج'))
                        st.write(f"**💰 السعر:** {metadata.get('price', 0):.2f} ريال")
                        st.write(f"**📝 الفئة:** {metadata.get('category', 'غير محدد')}")
                        
                        # Description (shorter for image results)
                        description = metadata.get('description', '')
                        if len(description) > 100:
                            description = description[:100] + "..."
                        st.write(f"**🔍 الوصف:** {description}")
                        
                        # Match score
                        match_percentage = result.score * 100
                        st.write(f"**🎯 درجة التشابه:** {match_percentage:.1f}%")
                        
                        # Add to cart button
                        if st.button(f"🛒 أضف للسلة", key=f"img_cart_{result.id}"):
                            st.success("تم إضافة المنتج للسلة! 🛍️")
                else:
                    st.warning("لم يتم العثور على منتجات مشابهة. جرب صورة أخرى.")

# Sidebar with helpful tips
st.sidebar.title("💡 نصائح للبحث")
st.sidebar.markdown("""
**للبحث الأفضل:**
- استخدم أوصاف واضحة (مثل: "خاتم ذهبي")
- اذكر الشكل أو التصميم (مثل: "بشكل الياسمين")
- اذكر المادة (ذهب، فضة، ماس)
- اذكر المناسبة (زواج، خطوبة، يومي)

**أمثلة بحث جيدة:**
- قلادة فضية بتصميم الأزهار
- خاتم خطوبة بحجر الماس
- أقراط ذهبية للمناسبات
- أساور بسيطة للاستخدام اليومي
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**🛍️ سلة التسوق**")
st.sidebar.write("العناصر المحددة: 0")
st.sidebar.button("عرض السلة")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    💎 مساعد متجر المجوهرات - مدعوم بالذكاء الاصطناعي
    </div>
    """, 
    unsafe_allow_html=True
)