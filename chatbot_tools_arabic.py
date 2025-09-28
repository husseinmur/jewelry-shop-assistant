import streamlit as st
from PIL import Image
import json
from shared.config import init_apis
from shared.langchain_rag import init_langchain_rag
from shared.embeddings import get_image_description
from shared.database import search_by_image
import openai

# Page config
st.set_page_config(
    page_title="مساعد متجر المجوهرات الذكي - Tools",
    page_icon="💎",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "مرحباً! أنا مساعدك الذكي لمتجر المجوهرات 💎\n\n🤖 **مدعوم بـ Function Calling**\n\nأستطيع أن أقرر متى أبحث في المخزون وأجيب على استفساراتك بذكاء.\n\nاسألني أي شيء عن المجوهرات!"
    })

if "rag_system" not in st.session_state:
    st.session_state.rag_system = None

# Initialize APIs and RAG
try:
    openai_client, pinecone_index = init_apis()

    if st.session_state.rag_system is None:
        with st.spinner("تحضير نظام البحث الذكي..."):
            st.session_state.rag_system = init_langchain_rag(
                pinecone_index,
                st.secrets["OPENAI_API_KEY"]
            )

        if st.session_state.rag_system:
            st.success("🚀 نظام البحث الذكي جاهز!")
        else:
            st.error("❌ فشل في تهيئة نظام البحث")
            st.stop()

except Exception as e:
    st.error(f"خطأ في الاتصال: {e}")
    st.stop()

st.title("💎 مساعد متجر المجوهرات الذكي")
st.markdown("### 🤖 مدعوم بـ Function Calling - الذكاء الاصطناعي يقرر متى يبحث")

# Tool definitions for the LLM
search_tool = {
    "type": "function",
    "function": {
        "name": "search_jewelry_products",
        "description": "البحث في مخزون المجوهرات في المتجر. استخدم هذه الأداة عندما يطلب العميل منتجات محددة أو يسأل عن ما متوفر في المتجر.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "نص البحث عن المجوهرات (مثل: سلاسل ذهبية، خواتم فضية، أقراط للزفاف)"
                }
            },
            "required": ["query"]
        }
    }
}

def search_jewelry_products(query: str, conversation_history: list = None) -> str:
    """Search for jewelry products and return formatted results"""
    try:
        if st.session_state.rag_system:
            _, results = st.session_state.rag_system.conversational_search(query, conversation_history)

            if results:
                # Format results for LLM context
                products_info = f"تم العثور على {len(results)} منتج في المخزون:\n\n"
                for i, result in enumerate(results, 1):
                    metadata = result['metadata']
                    products_info += f"{i}. {metadata.get('name', 'منتج')}\n"
                    products_info += f"   السعر: {metadata.get('price', 0):.2f} ريال\n"
                    products_info += f"   الفئة: {metadata.get('category', 'غير محدد')}\n"
                    if metadata.get('karat'):
                        products_info += f"   العيار: {metadata.get('karat')}\n"
                    if metadata.get('weight', 0) > 0:
                        products_info += f"   الوزن: {metadata.get('weight')} جرام\n"
                    if metadata.get('design'):
                        products_info += f"   التصميم: {metadata.get('design')}\n"
                    if metadata.get('product_url'):
                        products_info += f"   الرابط: {metadata.get('product_url')}\n"
                    products_info += f"   الوصف: {metadata.get('description', '')[:150]}...\n\n"

                # Store results - but don't always show them as cards
                st.session_state.last_search_results = results

                # Add instruction for LLM
                products_info += "\nتعليمات: تحدث بأسلوب دافئ ومرحب وودود. اذكر هذه المنتجات في إجابتك مع الأسعار والتفاصيل المهمة. تأكد من إدراج الرابط إذا كان متوفراً. استخدم عبارات ترحيبية وكن متحمساً لمساعدة العميل. تذكر: أنت تحافظ على سياق المحادثة وتربط إجابتك بما تم مناقشته سابقاً."

                return products_info
            else:
                return "أعتذر، لم أجد منتجات تطابق طلبك في مجموعتنا الحالية. لكن لا تقلق! يمكنني مساعدتك في البحث عن شيء آخر أو تقديم اقتراحات بديلة. ما رأيك أن نجرب بحثاً مختلفاً؟ 😊"
        else:
            return "نظام البحث غير متاح حالياً."

    except Exception as e:
        return f"حدث خطأ في البحث: {e}"

def get_ai_response_with_tools(user_message: str, conversation_history: list) -> tuple:
    """Get AI response with access to search tools"""
    try:
        # Prepare messages for the AI
        messages = [
            {
                "role": "system",
                "content": """أنت مساعد مبيعات ودود ومتحمس في متجر مجوهرات! 💎

شخصيتك:
- مرحب وودود دائماً
- متحمس لمساعدة العملاء
- خبير في المجوهرات ومتفهم لاحتياجات العميل
- تستخدم لغة دافئة ومشجعة

قواعد مهمة:
1. لديك إمكانية الوصول لأداة البحث في مخزون المتجر
2. استخدم أداة البحث فقط عندما يطلب العميل منتجات محددة أو يسأل عن المتوفر
3. للأسئلة العامة عن المجوهرات أو النصائح، أجب مباشرة بدون استخدام الأداة
4. عندما تحصل على نتائج البحث، اذكر المنتجات في إجابتك بطريقة طبيعية ومتحمسة
5. اذكر الأسماء والأسعار والتفاصيل المهمة في النص
6. ابدأ محادثاتك بترحيب دافئ واختتمها بعرض مساعدة إضافية
7. تحدث بالعربية دائماً واستخدم عبارات ودودة
8. 🔗 CRITICAL: تذكر المحادثة السابقة دائماً - راجع الرسائل السابقة واربط إجاباتك بما ناقشناه. إذا سأل عن "السعر" أو "الألوان" بدون تحديد، اربطه بآخر منتج ذكرناه
9. إذا قال العميل "ما السعر؟" بعد أن تحدثتم عن منتج معين، أجب عن سعر ذلك المنتج تحديداً

إرشاد للعرض:
- إذا كان العميل يتصفح أو يقارن، أضف: [SHOW_PRODUCTS] في نهاية إجابتك
- للاستفسارات البسيطة، لا تضع [SHOW_PRODUCTS]

أمثلة:
- "عندكن سلاسل؟" → اذكر السلاسل المتوفرة في النص
- "أريد أقارن بين الخواتم" → اذكر الخواتم + [SHOW_PRODUCTS]
- "كم سعر هذا الخاتم؟" → اذكر السعر في النص فقط
"""
            }
        ]

        # Add conversation history (last 6 messages)
        recent_history = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
        st.write(f"🔍 DEBUG: Processing {len(recent_history)} history messages")
        for msg in recent_history:
            if msg["role"] in ["user", "assistant"]:
                content = msg["content"]
                if len(content) > 200:
                    content = content[:200] + "..."
                messages.append({"role": msg["role"], "content": content})
                st.write(f"📝 Added to context: {msg['role']}: {content[:50]}...")

        # Add current user message
        messages.append({"role": "user", "content": user_message})

        # Call OpenAI with function calling
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            tools=[search_tool],
            tool_choice="auto",  # Let AI decide when to use tools
            temperature=0.3
        )

        response_message = response.choices[0].message
        search_results = None

        # Check if AI wants to use the search tool
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                if tool_call.function.name == "search_jewelry_products":
                    # Extract search query
                    function_args = json.loads(tool_call.function.arguments)
                    search_query = function_args.get("query", "")

                    # Perform search
                    search_result = search_jewelry_products(search_query, conversation_history)

                    # Add tool result to conversation
                    messages.append(response_message)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": search_result
                    })

                    # Get final response with search results
                    final_response = openai.chat.completions.create(
                        model="gpt-4",
                        messages=messages,
                        temperature=0.3
                    )

                    # Get final response content
                    final_content = final_response.choices[0].message.content

                    # Clean response content
                    clean_content = final_content.replace("[SHOW_PRODUCTS]", "").strip()

                    # Disable product cards completely for now
                    return clean_content, None

        # No tool calls - return direct response
        return response_message.content, None

    except Exception as e:
        return f"عذراً، حدث خطأ: {e}", None

def display_products(products):
    """Display product results"""
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

                # Additional details
                details = []
                if metadata.get('karat'):
                    details.append(f"العيار: {metadata.get('karat')}")
                if metadata.get('weight', 0) > 0:
                    details.append(f"الوزن: {metadata.get('weight')} جرام")
                if metadata.get('design'):
                    details.append(f"التصميم: {metadata.get('design')}")

                if details:
                    st.markdown(f"🔹 {' | '.join(details)}")

                # Description
                description = metadata.get('description', '')
                if len(description) > 100:
                    description = description[:100] + "..."
                st.markdown(f"📝 {description}")

                # Score
                if 'score' in result:
                    st.markdown(f"🎯 تطابق: {result['score'] * 100:.1f}%")

                # Add to cart
                if st.button(f"🛒 أضف للسلة", key=f"cart_{result['id']}_{idx}"):
                    st.success("تم إضافة المنتج للسلة! 🛍️")

                st.markdown("---")

# Chat interface
st.markdown("---")

# Display chat history (no product cards)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Product cards disabled completely
        # if message["role"] == "assistant" and "products" in message:
        #     if message["products"]:
        #         st.markdown("### 🛍️ منتجات مقترحة:")
        #         display_products(message["products"])

# Image upload (keep existing functionality)
st.markdown("### 📸 رفع صورة (اختياري)")
uploaded_image = st.file_uploader(
    "ارفع صورة قطعة مجوهرات:",
    type=['png', 'jpg', 'jpeg']
)

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="الصورة المرفوعة", width=200)

    if st.button("🔍 تحليل الصورة"):
        with st.spinner("تحليل الصورة..."):
            description = get_image_description(image)
            search_results = search_by_image(pinecone_index, image, top_k=5)

            formatted_results = []
            if search_results:
                for result in search_results:
                    formatted_results.append({
                        'id': result.id,
                        'score': result.score,
                        'metadata': result.metadata
                    })

            analysis_query = f"حلل هذه الصورة: {description}"
            bot_response, _ = get_ai_response_with_tools(analysis_query, st.session_state.messages)

            st.session_state.messages.append({
                "role": "user",
                "content": "🖼️ رفعت صورة قطعة مجوهرات"
            })
            st.session_state.messages.append({
                "role": "assistant",
                "content": bot_response
            })

            st.rerun()

# Chat input
if prompt := st.chat_input("اكتب رسالتك هنا..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response with tools
    with st.chat_message("assistant"):
        with st.spinner("أفكر..."):
            response, search_results = get_ai_response_with_tools(prompt, st.session_state.messages)

            st.markdown(response)

            # Product cards disabled - all info in conversational text
            # if search_results:
            #     st.markdown("### 🛍️ منتجات مقترحة:")
            #     display_products(search_results)

    # Add assistant response to history (no product cards)
    assistant_message = {
        "role": "assistant",
        "content": response
    }

    st.session_state.messages.append(assistant_message)

# Sidebar
st.sidebar.title("🤖 Function Calling")
st.sidebar.markdown("""
**كيف يعمل:**

🧠 **AI يقرر كل شيء:**
- متى يبحث في المخزون
- كيف يعرض النتائج
- متى يظهر كروت المنتجات

💬 **عرض محادثة فقط:**
- يدمج المنتجات في النص طبيعياً
- لا توجد كروت منتجات
- كل المعلومات في الحديث

**أمثلة:**

🔍 **"عندكن سلاسل؟"**
→ سيبحث ويذكر السلاسل في النص

🛒 **"أريد أقارن الخواتم"**
→ سيبحث ويذكر الخواتم مع المقارنة

💰 **"كم سعر السلسلة الذهبية؟"**
→ سيذكر السعر في النص فقط

💬 **"مرحباً"**
→ رد مباشر بدون بحث
""")

st.sidebar.markdown("---")
if st.sidebar.button("🧹 مسح المحادثة"):
    # Clear everything including any cached product data
    st.session_state.clear()
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "تم مسح المحادثة! 🧹\n\nمرحباً مجدداً! اسألني أي شيء عن المجوهرات."
    })
    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    💎 مساعد متجر المجوهرات الذكي<br>
    🤖 مدعوم بـ Function Calling - الذكاء الاصطناعي يقرر
    </div>
    """,
    unsafe_allow_html=True
)