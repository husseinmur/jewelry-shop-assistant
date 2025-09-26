import streamlit as st
from PIL import Image
import uuid
import os
from shared.config import init_apis
from shared.database import store_product, get_all_products, delete_product

# Page config
st.set_page_config(
    page_title="متجر المجوهرات - الإدارة",
    page_icon="💎",
    layout="wide"
)

st.title("💎 متجر المجوهرات - إدارة المنتجات")
st.sidebar.title("لوحة الإدارة")

# Initialize APIs
try:
    openai_client, pinecone_index = init_apis()
    st.sidebar.success("✅ متصل بقواعد البيانات")
except Exception as e:
    st.sidebar.error(f"❌ خطأ في الاتصال: {e}")
    st.stop()

# Sidebar navigation
page = st.sidebar.selectbox("اختر الإجراء:", ["إضافة منتجات", "عرض المنتجات", "رفع مجمع"])

if page == "إضافة منتجات":
    st.header("إضافة منتج جديد")
    
    # Upload image
    uploaded_file = st.file_uploader(
        "رفع صورة المنتج",
        type=['png', 'jpg', 'jpeg'],
        help="ارفع صورة واضحة لقطعة المجوهرات"
    )
    
    if uploaded_file:
        # Display image
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image, caption="صورة المنتج", use_container_width=True)
        
        with col2:
            # Product details form
            with st.form("product_form"):
                name = st.text_input(
                    "اسم المنتج*",
                    placeholder="مثال: أقراط بتلات الورد الفضية"
                )
                
                price = st.number_input(
                    "السعر (ريال)*",
                    min_value=0.01,
                    step=0.01,
                    format="%.2f"
                )
                
                category = st.selectbox(
                    "الفئة*",
                    ["خواتم", "عقود", "أقراط", "أساور", "دبابيس", "طقم", "أخرى"]
                )
                
                # Additional jewelry-specific fields
                st.subheader("تفاصيل المجوهرات")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    karat = st.selectbox(
                        "العيار",
                        ["", "18 قيراط", "21 قيراط", "24 قيراط", "فضة 925", "فضة 999", "بلاتين", "أخرى"],
                        help="اختر عيار المعدن"
                    )
                    
                    design = st.text_input(
                        "التصميم",
                        placeholder="مثال: زهرة الياسمين، هندسي، كلاسيكي...",
                        help="نوع أو شكل التصميم"
                    )
                
                with col_b:
                    weight = st.number_input(
                        "الوزن (جرام)",
                        min_value=0.0,
                        step=0.1,
                        format="%.1f",
                        help="وزن القطعة بالجرام"
                    )
                    
                    style = st.selectbox(
                        "الستايل",
                        ["", "عصري", "كلاسيكي", "عتيق", "بسيط", "فاخر", "رومانسي", "هندسي", "بوهيمي", "أخرى"],
                        help="الطراز العام للقطعة"
                    )

                # URL field
                st.subheader("رابط المنتج")
                product_url = st.text_input(
                    "رابط URL (اختياري)",
                    placeholder="https://example.com/product/...",
                    help="رابط المنتج على الموقع أو متجر إلكتروني"
                )

                # Additional description
                st.subheader("وصف إضافي")
                additional_description = st.text_area(
                    "تفاصيل إضافية (اختياري)",
                    placeholder="أضف أي تفاصيل إضافية حول المواد، الأحجار الكريمة، الطراز، الحرفية، إلخ",
                    height=100,
                    help="سيتم دمج هذا مع الوصف المُولد بالذكاء الاصطناعي لتحسين نتائج البحث"
                )
                
                submitted = st.form_submit_button("إضافة المنتج", type="primary")
                
                if submitted:
                    if name and price:
                        with st.spinner("معالجة المنتج..."):
                            # Save image temporarily
                            image_filename = f"product_{uuid.uuid4()}.jpg"
                            
                            # Store product
                            success = store_product(
                                index=pinecone_index,
                                image=image,
                                name=name,
                                price=price,
                                category=category,
                                image_url=f"images/{image_filename}",
                                additional_info=additional_description,
                                karat=karat,
                                weight=weight,
                                design=design,
                                style=style,
                                product_url=product_url
                            )
                            
                            if success:
                                st.success(f"✅ تم إضافة '{name}' إلى الكتالوج!")
                                st.balloons()
                    else:
                        st.error("يرجى ملء الحقول المطلوبة (الاسم والسعر)")

elif page == "عرض المنتجات":
    st.header("كتالوج المنتجات")
    
    # Fetch products
    with st.spinner("تحميل المنتجات..."):
        products = get_all_products(pinecone_index, limit=50)
    
    if products:
        st.write(f"تم العثور على {len(products)} منتج")
        
        # Display products in grid
        cols = st.columns(3)
        
        for idx, product in enumerate(products):
            with cols[idx % 3]:
                metadata = product.metadata
                
                # Product card
                st.subheader(metadata.get('name', 'منتج بدون اسم'))
                st.write(f"**السعر:** {metadata.get('price', 0):.2f} ريال")
                st.write(f"**الفئة:** {metadata.get('category', 'غير معروف')}")
                
                # Show jewelry details
                if metadata.get('karat'):
                    st.write(f"**العيار:** {metadata.get('karat')}")
                if metadata.get('weight', 0) > 0:
                    st.write(f"**الوزن:** {metadata.get('weight')} جرام")
                if metadata.get('design'):
                    st.write(f"**التصميم:** {metadata.get('design')}")
                if metadata.get('style'):
                    st.write(f"**الستايل:** {metadata.get('style')}")
                if metadata.get('product_url'):
                    st.write(f"**الرابط:** [عرض المنتج]({metadata.get('product_url')})")

                # Description (truncated)
                description = metadata.get('description', '')
                if len(description) > 100:
                    description = description[:100] + "..."
                st.write(f"**الوصف:** {description}")
                
                # Similarity score
                st.write(f"**درجة التطابق:** {product.score:.3f}")
                
                # Delete button
                if st.button(f"🗑️ حذف", key=f"delete_{product.id}"):
                    if delete_product(pinecone_index, product.id):
                        st.rerun()
                
                st.divider()
    else:
        st.info("لم يتم العثور على منتجات. أضف بعض المنتجات أولاً!")

elif page == "رفع مجمع":
    st.header("رفع منتجات مجمع")
    st.info("ارفع عدة صور منتجات دفعة واحدة")
    
    uploaded_files = st.file_uploader(
        "اختر صور المنتجات",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True,
        help="اختر عدة صور مجوهرات للرفع"
    )
    
    if uploaded_files:
        st.write(f"تم اختيار {len(uploaded_files)} ملف")
        
        # Default category for bulk upload
        bulk_category = st.selectbox(
            "الفئة الافتراضية لجميع العناصر:",
            ["خواتم", "عقود", "أقراط", "أساور", "دبابيس", "طقم", "أخرى"]
        )
        
        if st.button("معالجة جميع الصور", type="primary"):
            progress_bar = st.progress(0)
            success_count = 0
            
            for idx, uploaded_file in enumerate(uploaded_files):
                try:
                    # Update progress
                    progress = (idx + 1) / len(uploaded_files)
                    progress_bar.progress(progress)
                    
                    # Process image
                    image = Image.open(uploaded_file)
                    filename_without_ext = os.path.splitext(uploaded_file.name)[0]
                    
                    # Use filename as product name
                    product_name = filename_without_ext.replace('_', ' ').replace('-', ' ').title()
                    
                    # Default price
                    default_price = 99.99
                    
                    # Store product
                    success = store_product(
                        index=pinecone_index,
                        image=image,
                        name=product_name,
                        price=default_price,
                        category=bulk_category,
                        image_url=f"images/{uploaded_file.name}",
                        additional_info="",
                        karat="",
                        weight=0.0,
                        design="",
                        style="",
                        product_url=""
                    )
                    
                    if success:
                        success_count += 1
                        
                except Exception as e:
                    st.error(f"خطأ في معالجة {uploaded_file.name}: {e}")
            
            st.success(f"✅ تم معالجة {success_count}/{len(uploaded_files)} منتج بنجاح!")
            st.info("💡 نصيحة: استخدم 'عرض المنتجات' لمراجعة وتعديل التفاصيل المُولدة تلقائياً.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**أدوات الإدارة**")
if st.sidebar.button("🔄 تحديث اتصال قاعدة البيانات"):
    st.rerun()

st.sidebar.markdown("**إحصائيات سريعة**")
try:
    all_products = get_all_products(pinecone_index, limit=1000)
    st.sidebar.metric("إجمالي المنتجات", len(all_products))
except:
    st.sidebar.metric("إجمالي المنتجات", "خطأ")