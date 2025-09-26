import streamlit as st
from PIL import Image
import json
from datetime import datetime
from shared.config import init_apis, TEXT_MODEL
from shared.langchain_rag import init_langchain_rag
from shared.embeddings import get_image_description, get_image_category
from shared.database import search_by_image  # Keep for image search
import openai

# Page config
st.set_page_config(
    page_title="مساعد متجر المجوهرات",
    page_icon="💎",
    layout="wide"
)

# Add RTL CSS and chat input positioning
st.markdown("""
<style>
    .main .block-container {
        direction: rtl;
        text-align: right;
        padding-bottom: 100px;
    }
    .stChatMessage {
        direction: rtl;
        text-align: right;
    }
    .stTextInput > div > div > input {
        direction: rtl;
        text-align: right;
    }
    .stFileUploader > div {
        direction: rtl;
        text-align: right;
    }
    h1 {
        direction: rtl;
        text-align: right;
    }
    .stRadio {
        direction: rtl !important;
        text-align: right !important;
    }
    .stRadio > div[role="radiogroup"] {
        direction: rtl !important;
        justify-content: flex-end !important;
        display: flex !important;
        flex-direction: row-reverse !important;
    }
    .stRadio > div[role="radiogroup"] > label {
        direction: rtl !important;
        margin-left: 1rem !important;
        margin-right: 0 !important;
    }
    .stRadio div[data-testid="stHorizontalBlock"] {
        direction: rtl !important;
        justify-content: flex-end !important;
    }
    .stChatInputContainer {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        z-index: 1000;
        padding: 10px 20px;
        border-top: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add initial assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": "مرحباً! كيف يمكنني مساعدتك؟"
    })

if "rag_system" not in st.session_state:
    st.session_state.rag_system = None

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "chat"



# Initialize APIs and RAG
try:
    openai_client, pinecone_index = init_apis()

    # Initialize LangChain RAG system
    if st.session_state.rag_system is None:
        with st.spinner("تحضير نظام البحث الذكي..."):
            st.session_state.rag_system = init_langchain_rag(
                pinecone_index,
                st.secrets["OPENAI_API_KEY"]
            )

        if not st.session_state.rag_system:
            st.error("❌ فشل في تهيئة نظام البحث")
            st.stop()

except Exception as e:
    st.error(f"خطأ في الاتصال: {e}")
    st.stop()

st.title("💎 مساعد متجر المجوهرات")

def is_product_query(message: str) -> bool:
    """
    More precise detection of product queries
    Only search when user is clearly looking for products
    """
    message_lower = message.lower()

    # Explicit product requests - these should definitely trigger search
    explicit_requests = [
        "أريد", "ابحث", "عندكن", "عندكم", "متوفر", "موجود", "لديكم", "لديكن",
        "اعرض", "أعرض", "وريني", "أوريني", "اطلب", "أطلب"
    ]

    # Product names - only search if mentioned
    product_names = [
        "خاتم", "خواتم", "عقد", "عقود", "قلادة", "قلائد",
        "سلسلة", "سلاسل", "سلسال", "أقراط", "قرط",
        "سوار", "أساور", "اسورة", "دبوس", "دبابيس", "طقم", "أطقم"
    ]

    # Materials with intent words
    material_queries = [
        "ذهب", "ذهبي", "ذهبية", "فضة", "فضي", "فضية",
        "ماس", "ألماس", "لؤلؤ"
    ]

    # Check for explicit requests
    has_explicit_request = any(req in message_lower for req in explicit_requests)

    # Check for product names
    has_product_name = any(product in message_lower for product in product_names)

    # Check for material queries with some context
    has_material_context = any(material in message_lower for material in material_queries)

    # Product questions with question words
    question_indicators = ["ما المتوفر", "ماذا عندكم", "ماذا لديكم", "ما عندكن", "ما لديكن"]
    has_product_question = any(q in message_lower for q in question_indicators)

    # Only trigger search if:
    # 1. Explicit request OR
    # 2. Product name mentioned OR
    # 3. Material mentioned with some context OR
    # 4. Specific product question
    return (has_explicit_request or
            has_product_name or
            (has_material_context and len(message.split()) > 2) or
            has_product_question)

def get_general_response(prompt: str) -> str:
    """Handle general conversation that doesn't require product search"""
    try:
        response = openai.chat.completions.create(
            model=TEXT_MODEL,
            messages=[
                {"role": "system", "content": """أنت مساعد ودود لمتجر مجوهرات.

                للأسئلة العامة عن المجوهرات (كالعناية، المناسبات، الأنواع)، قدم نصائح مفيدة.
                للاستفسارات عن منتجات محددة، انصح العميل بأن يسأل عن منتجات معينة ليمكنني البحث في مخزوننا.

                كن مفيداً وودوداً وتحدث بالعربية."""},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"عذراً، حدث خطأ: {e}"

def display_products(products):
    """Display product results in a nice format"""
    if not products:
        return

    cols = st.columns(min(len(products), 3))
    for idx, result in enumerate(products):
        with cols[idx % 3]:
            metadata = result['metadata']

            with st.container():
                st.markdown(f"**{metadata.get('name', 'منتج')}**")
                st.markdown(f"💰 **{metadata.get('price', 0):.2f} ريال**")
                st.markdown(f"📂 {metadata.get('category', 'غير محدد')}")

                # Show additional details
                details = []
                if metadata.get('karat'):
                    details.append(f"العيار: {metadata.get('karat')}")
                if metadata.get('weight', 0) > 0:
                    details.append(f"الوزن: {metadata.get('weight')} جرام")
                if metadata.get('design'):
                    details.append(f"التصميم: {metadata.get('design')}")

                if details:
                    st.markdown(f"🔹 {' | '.join(details)}")

                # Description (truncated)
                description = metadata.get('description', '')
                if len(description) > 100:
                    description = description[:100] + "..."
                st.markdown(f"📝 {description}")

                # Match score
                if 'score' in result:
                    st.markdown(f"🎯 تطابق: {result['score'] * 100:.1f}%")

                # Add to cart button
                if st.button(f"🛒 أضف للسلة", key=f"cart_{result['id']}_{idx}"):
                    st.success("تم إضافة المنتج للسلة! 🛍️")

                st.markdown("---")

# Tab selection with radio buttons - using columns to align right
col1, col2 = st.columns([3, 1])

with col2:
    tab_choice = st.radio(
        "اختر النشاط:",
        ["📸 بحث بالصورة", "💬 محادثة نصية"],
        horizontal=True,
        index=1  # Default to "محادثة نصية" (text search)
    )

st.session_state.active_tab = "chat" if "محادثة" in tab_choice else "image"

if st.session_state.active_tab == "chat":
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

elif st.session_state.active_tab == "image":
    # Image upload section
    uploaded_image = st.file_uploader(
        "اختر صورة قطعة مجوهرات:",
        type=['png', 'jpg', 'jpeg'],
        help="يمكنك رفع صورة قطعة مجوهرات لأحصل على معلومات عنها أو أجد قطع مشابهة"
    )

    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, caption="الصورة المرفوعة", width=300)

        if st.button("🔍 تحليل الصورة وابحث عن مشابهة", type="primary"):
            with st.spinner("تحليل الصورة والبحث..."):
                # Analyze the image
                description = get_image_description(image)
                category = get_image_category(image)

                # Search for similar products using traditional image search
                search_results = search_by_image(pinecone_index, image, top_k=5)

                # Convert results to our format for display
                formatted_results = []
                if search_results:
                    for result in search_results:
                        formatted_results.append({
                            'id': result.id,
                            'score': result.score,
                            'metadata': result.metadata
                        })

                # Generate response using RAG
                analysis_query = f"حلل هذه الصورة: {description}. الفئة المكتشفة: {category}"
                if st.session_state.rag_system:
                    bot_response, _ = st.session_state.rag_system.conversational_search(
                        analysis_query
                    )
                else:
                    bot_response = f"تم تحليل الصورة: {description}\nالفئة: {category}"

                # Display results
                st.markdown(bot_response)

                if formatted_results:
                    display_products(formatted_results)
                else:
                    st.info("لم يتم العثور على منتجات مشابهة")

                # Add to chat history
                st.session_state.messages.append({
                    "role": "user",
                    "content": "🖼️ رفعت صورة قطعة مجوهرات"
                })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": bot_response,
                    "products": formatted_results if formatted_results else None
                })
    else:
        st.info("اختر صورة")

# Chat input - only show on chat tab
if st.session_state.active_tab == "chat":
    if prompt := st.chat_input("اكتب رسالتك هنا..."):
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant thinking and response
        with st.chat_message("assistant"):
            thinking_placeholder = st.empty()
            thinking_placeholder.markdown("🤔 أفكر...")

            if st.session_state.rag_system and is_product_query(prompt):
                response, search_results = st.session_state.rag_system.conversational_search(
                    prompt,
                    conversation_history=st.session_state.messages
                )
            else:
                response = get_general_response(prompt)

            thinking_placeholder.markdown(response)

        # Add to history after display
        st.session_state.messages.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ])

