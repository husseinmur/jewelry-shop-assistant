import streamlit as st
from PIL import Image
import json
from datetime import datetime
from shared.config import init_apis, TEXT_MODEL
from shared.database import search_by_text, search_by_image, smart_search
from shared.embeddings import get_image_description
import openai

# Page config
st.set_page_config(
    page_title="مساعد متجر المجوهرات الذكي",
    page_icon="💎",
    layout="wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add initial assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": "مرحباً! أنا مساعدك الذكي لمتجر المجوهرات 💎\n\nيمكنني مساعدتك في:\n• البحث عن قطع مجوهرات محددة\n• تقديم نصائح حول اختيار المجوهرات\n• شرح المواد والعيارات\n• اقتراح قطع مناسبة للمناسبات\n• الإجابة على أسئلتك حول المنتجات\n\nكيف يمكنني مساعدتك اليوم؟"
    })

if "awaiting_image" not in st.session_state:
    st.session_state.awaiting_image = False

# Initialize APIs
try:
    openai_client, pinecone_index = init_apis()
except Exception as e:
    st.error(f"خطأ في الاتصال: {e}")
    st.stop()

st.title("💎 مساعد متجر المجوهرات الذكي")
st.markdown("### محادثة ذكية مع خبير المجوهرات")

def get_chatbot_response(user_message, search_results=None, image_analysis=None):
    """Generate chatbot response using OpenAI with strict RAG enforcement"""
    try:
        # Always search for products if we don't have search results already
        if not search_results and should_search_products(user_message):
            search_results = smart_search(pinecone_index, user_message, search_type="text", top_k=5)

        # Build strict RAG system message
        if search_results and len(search_results) > 0:
            # RAG mode: Only use database knowledge
            system_message = """أنت مساعد مبيعات ودود ومرحب في متجر مجوهرات! 💎

            شخصيتك:
            - مرحب وودود جداً مع كل عميل
            - متحمس لمساعدة العملاء في العثور على أجمل القطع
            - تستخدم لغة دافئة ومحببة
            - تبدي اهتماماً حقيقياً بإسعاد العميل

            🌟 قواعد مهمة:
            1. استخدم فقط المعلومات المتوفرة في قاعدة بيانات المتجر أدناه
            2. لا تخترع أو تضيف معلومات من معرفتك العامة
            3. إذا لم تجد معلومة في قاعدة البيانات، قل "للأسف هذه المعلومة غير متوفرة حالياً" بطريقة ودودة
            4. اربط إجابتك دائماً بالمنتجات الموجودة في المتجر بحماس
            5. تأكد من إدراج الرابط إذا كان متوفراً للمنتج
            6. ابدأ بترحيب دافئ واختتم بعرض مساعدة إضافية
            7. استخدم عبارات مثل "يسعدني أن أساعدك" و "أتمنى أن تنال إعجابك"

            تحدث باللغة العربية بأسلوب ودود ومتحمس!"""

            # Add product database
            products_info = "🏪 قاعدة بيانات منتجات المتجر:\n\n"
            for idx, result in enumerate(search_results[:5], 1):
                metadata = result.metadata
                products_info += f"منتج #{idx}: {metadata.get('name', 'منتج')}\n"
                products_info += f"- السعر: {metadata.get('price', 0):.2f} ريال\n"
                products_info += f"- الفئة: {metadata.get('category', 'غير محدد')}\n"
                if metadata.get('karat'):
                    products_info += f"- العيار: {metadata.get('karat')}\n"
                if metadata.get('weight', 0) > 0:
                    products_info += f"- الوزن: {metadata.get('weight')} جرام\n"
                if metadata.get('design'):
                    products_info += f"- التصميم: {metadata.get('design')}\n"
                if metadata.get('style'):
                    products_info += f"- الستايل: {metadata.get('style')}\n"
                if metadata.get('product_url'):
                    products_info += f"- الرابط: {metadata.get('product_url')}\n"
                products_info += f"- الوصف: {metadata.get('description', '')}\n"
                products_info += f"- مدى التطابق مع البحث: {result.score * 100:.1f}%\n\n"

            products_info += "\n💫 هذه هي مجموعة منتجاتنا الرائعة المتطابقة مع طلبك. استخدم هذه المعلومات فقط وكن متحمساً في عرضها!"

        else:
            # No products found or general question
            system_message = """أنت مساعد مبيعات ودود ومتحمس في متجر مجوهرات! 💎

            الموقف الحالي:
            - لم نجد منتجات تطابق تماماً طلب العميل في مجموعتنا الحالية
            - أو أن العميل يطرح سؤالاً عاماً عن المجوهرات

            شخصيتك الودودة:
            - اعتذر بلطف ودود عن عدم وجود المنتج المحدد
            - أظهر تفهماً واهتماماً بحاجة العميل
            - اقترح بدائل أو طرق بحث مختلفة بحماس
            - قدم نصائح مفيدة حول المجوهرات والعناية بها
            - شجع العميل على تصفح منتجاتنا الرائعة الأخرى

            🌟 أسلوبك:
            - استخدم عبارات مثل "يسعدني مساعدتك" و "أتفهم تماماً ما تبحث عنه"
            - كن متفائلاً ومشجعاً
            - اختتم بعرض مساعدة إضافية

            تحدث بالعربية بأسلوب دافئ ومرحب!"""
            products_info = "💝 هذه فرصة رائعة لتقديم المساعدة والنصائح حول المجوهرات بأسلوب ودود!"

        messages = [
            {"role": "system", "content": system_message},
            {"role": "system", "content": products_info}
        ]

        # Add recent conversation history (only last 4 to save tokens)
        recent_messages = st.session_state.messages[-4:] if len(st.session_state.messages) > 4 else st.session_state.messages
        for msg in recent_messages:
            if msg["role"] in ["user", "assistant"]:
                # Truncate long messages to save tokens
                content = msg["content"]
                if len(content) > 200:
                    content = content[:200] + "..."
                messages.append({"role": msg["role"], "content": content})

        # Add image analysis if available
        if image_analysis:
            messages.append({"role": "system", "content": f"📸 تحليل الصورة المرفوعة: {image_analysis}"})

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        response = openai.chat.completions.create(
            model=TEXT_MODEL,
            messages=messages,
            max_tokens=400,
            temperature=0.3  # Lower temperature for more consistent responses
        )

        return response.choices[0].message.content, search_results

    except Exception as e:
        return f"عذراً، حدث خطأ في معالجة طلبك: {e}", None

def should_search_products(message):
    """Determine if the user message requires product search"""
    search_keywords = [
        # Intent words
        "ابحث", "أبحث", "أريد", "أطلب", "اعرض", "أعرض", "وريني", "أوريني",
        "عندكن", "عندكم", "متوفر", "موجود", "يوجد", "عرضوا", "لديكم", "لديكن",

        # Products (singular and plural)
        "خاتم", "خواتم", "عقد", "عقود", "قلادة", "قلائد", "سلسلة", "سلاسل",
        "أقراط", "قرط", "أساور", "سوار", "اسورة", "دبوس", "دبابيس", "طقم", "أطقم",

        # Materials
        "ذهب", "ذهبي", "ذهبية", "فضة", "فضي", "فضية", "ماس", "ألماس",
        "لؤلؤ", "زمرد", "ياقوت", "بلاتين",

        # Occasions and style
        "للزفاف", "للخطوبة", "للمناسبة", "هدية", "بتصميم", "بشكل",
        "عصري", "كلاسيكي", "بسيط", "فاخر", "أنيق"
    ]

    message_lower = message.lower()
    return any(keyword in message_lower for keyword in search_keywords)

# Chat interface
st.markdown("---")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Product cards disabled - all info in conversational text
        # if message["role"] == "assistant" and "products" in message:
        #     display_products(message["products"])

# Image upload section
st.markdown("### 📸 رفع صورة (اختياري)")
uploaded_image = st.file_uploader(
    "ارفع صورة قطعة مجوهرات للاستفسار عنها أو البحث عن مشابهة:",
    type=['png', 'jpg', 'jpeg'],
    help="يمكنك رفع صورة قطعة مجوهرات لأحصل على معلومات عنها أو أجد قطع مشابهة"
)

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="الصورة المرفوعة", width=200)

    if st.button("🔍 تحليل الصورة"):
        with st.spinner("تحليل الصورة..."):
            # Analyze the image
            description = get_image_description(image)

            # Search for similar products
            search_results = search_by_image(pinecone_index, image, top_k=5)

            # Generate chatbot response
            analysis_text = f"وصف القطعة: {description}"
            bot_response, _ = get_chatbot_response(
                "حلل هذه الصورة واعطني معلومات عنها",
                search_results=search_results,
                image_analysis=analysis_text
            )

            # Add to chat history
            st.session_state.messages.append({
                "role": "user",
                "content": "🖼️ رفعت صورة قطعة مجوهرات"
            })
            st.session_state.messages.append({
                "role": "assistant",
                "content": bot_response
                # Product cards disabled - not storing products
                # "products": search_results if search_results else None
            })

            st.rerun()

# Chat input
if prompt := st.chat_input("اكتب رسالتك هنا..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("أفكر..."):
            # Generate response (search is handled inside get_chatbot_response now)
            response, search_results = get_chatbot_response(prompt)

            st.markdown(response)

            # Product cards disabled
            # if search_results:
            #     st.markdown("### 🛍️ منتجات مقترحة:")
            #     display_products(search_results)

    # Add assistant response to chat history
    assistant_message = {
        "role": "assistant",
        "content": response
    }
    # Product cards disabled - not storing products in messages
    # if search_results:
    #     assistant_message["products"] = search_results

    st.session_state.messages.append(assistant_message)

def display_products(products):
    """Display product results in a nice format"""
    if not products:
        return

    cols = st.columns(min(len(products), 3))
    for idx, result in enumerate(products):
        with cols[idx % 3]:
            metadata = result.metadata

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
                if len(description) > 80:
                    description = description[:80] + "..."
                st.markdown(f"📝 {description}")

                # Match score
                st.markdown(f"🎯 تطابق: {result.score * 100:.1f}%")

                # Add to cart button
                if st.button(f"🛒 أضف للسلة", key=f"cart_{result.id}_{idx}"):
                    st.success("تم إضافة المنتج للسلة! 🛍️")

                st.markdown("---")

# Sidebar with tips and features
st.sidebar.title("💡 ميزات المساعد الذكي")
st.sidebar.markdown("""
**يمكنني مساعدتك في:**

🔍 **البحث والاستفسار:**
- "أريد خاتم ذهبي بسيط"
- "ما أفضل عقد للزفاف؟"
- "عرفني على أقراط اللؤلؤ"

💎 **النصائح والمشورة:**
- اختيار المجوهرات للمناسبات
- الفرق بين العيارات
- نصائح العناية بالمجوهرات

📸 **تحليل الصور:**
- ارفع صورة قطعة مجوهرات
- سأحللها وأجد قطع مشابهة
- سأعطيك معلومات عن التصميم

🎁 **اقتراح الهدايا:**
- "ما أفضل هدية لخطوبة؟"
- "اقترح مجوهرات للأم"
- "قطع مناسبة للاستخدام اليومي"
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**🛠️ أدوات سريعة**")

if st.sidebar.button("🧹 مسح المحادثة"):
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "تم مسح المحادثة! 🧹\n\nمرحباً مجدداً! كيف يمكنني مساعدتك في العثور على المجوهرات المثالية؟"
    })
    st.rerun()

if st.sidebar.button("💡 أمثلة أسئلة"):
    example_questions = [
        "أريد خاتم خطوبة بحجر الماس",
        "ما الفرق بين الذهب عيار 18 و 21؟",
        "اقترح لي عقد مناسب للعمل",
        "كيف أعتني بمجوهرات الفضة؟",
        "أريد طقم مجوهرات للزفاف"
    ]
    st.sidebar.markdown("**أمثلة يمكنك سؤالها:**")
    for q in example_questions:
        st.sidebar.markdown(f"• {q}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**📊 إحصائيات الجلسة**")
st.sidebar.metric("عدد الرسائل", len(st.session_state.messages))
st.sidebar.metric("وقت بدء الجلسة", datetime.now().strftime("%H:%M"))

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    💎 مساعد متجر المجوهرات الذكي - مدعوم بالذكاء الاصطناعي<br>
    💬 محادثة تفاعلية | 🔍 بحث ذكي | 📸 تحليل الصور
    </div>
    """,
    unsafe_allow_html=True
)