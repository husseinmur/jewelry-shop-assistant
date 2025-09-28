import streamlit as st
from PIL import Image
import json
from datetime import datetime
from shared.config import init_apis, TEXT_MODEL
from shared.langchain_rag import init_langchain_rag
from shared.embeddings import get_image_description
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

                # Add instruction for LLM
                products_info += "\nتعليمات: تحدث بأسلوب دافئ ومرحب وودود. اذكر هذه المنتجات في إجابتك مع الأسعار والتفاصيل المهمة. تأكد من إدراج الرابط إذا كان متوفراً. تذكر: أنت تحافظ على سياق المحادثة وتربط إجابتك بما تم مناقشته سابقاً."

                return products_info
            else:
                return "أعتذر، لم أجد منتجات تطابق طلبك في مجموعتنا الحالية. لكن لا تقلق! يمكنني مساعدتك في البحث عن شيء آخر أو تقديم اقتراحات بديلة. ما رأيك أن نجرب بحثاً مختلفاً؟ 😊"
        else:
            return "نظام البحث غير متاح حالياً."

    except Exception as e:
        return f"حدث خطأ في البحث: {e}"

def get_ai_response_with_tools(user_message: str, conversation_history: list) -> str:
    """Get AI response with access to search tools and full conversation context"""
    try:
        # Define the search tool
        search_tool = {
            "type": "function",
            "function": {
                "name": "search_jewelry_products",
                "description": "Search for jewelry products in the store inventory when customer asks about specific products or wants to see what's available",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for jewelry products"
                        }
                    },
                    "required": ["query"]
                }
            }
        }

        # Build conversation context summary FIRST
        context_summary = ""
        if conversation_history:
            recent_history = conversation_history[-4:] if len(conversation_history) > 4 else conversation_history
            if recent_history:
                context_summary = "📋 ملخص المحادثة السابقة:\n"
                for i, msg in enumerate(recent_history):
                    if msg["role"] == "user":
                        context_summary += f"العميل قال: {msg['content']}\n"
                    elif msg["role"] == "assistant":
                        context_summary += f"أنت أجبت: {msg['content'][:100]}...\n"

                # Extract key information
                mentioned_products = []
                for msg in recent_history:
                    content = msg['content'].lower()
                    if any(product in content for product in ['خاتم', 'عقد', 'سلسلة', 'أقراط', 'سوار']):
                        mentioned_products.append(msg['content'][:150])

                if mentioned_products:
                    context_summary += f"\n🎯 المنتجات التي تم ذكرها:\n"
                    for product in mentioned_products[-2:]:  # Last 2 product mentions
                        context_summary += f"- {product}\n"

        # Prepare messages with CONTEXT FIRST
        messages = [
            {
                "role": "system",
                "content": f"""أنت مساعد مبيعات ودود في متجر مجوهرات.

{context_summary}

🔗 CRITICAL: إذا سأل العميل عن "السعر" أو "الألوان" أو "متوفر" بدون تحديد المنتج،
يجب أن تربط السؤال بآخر منتج ذكرته في المحادثة أعلاه.

قواعد أساسية:
1. راجع المحادثة السابقة أولاً قبل أي شيء
2. اربط الأسئلة الغامضة بالسياق السابق
3. لديك أداة بحث للمنتجات الجديدة فقط
4. لا تخترع معلومات - استخدم فقط ما في البحث أو المحادثة
5. كن ودوداً ومتحمساً

🎯 مثال: إذا ذكرت خواتم سابقاً وسأل "كم السعر؟" → أجب عن أسعار الخواتم من المحادثة السابقة."""
            }
        ]

        # Add recent conversation history
        recent_history = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
        for msg in recent_history:
            if msg["role"] in ["user", "assistant"]:
                messages.append({"role": msg["role"], "content": msg["content"]})

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

                    return final_response.choices[0].message.content

        # No tool call needed, return direct response
        return response_message.content

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

                # Create a specific search query based on image details
                # Extract key features for better matching
                search_query = description  # Use the detailed image description directly

                # Use the search tool directly for better matching
                search_results = search_jewelry_products(search_query, st.session_state.messages)

                # Create analysis prompt with search results
                analysis_query = f"""لقد رفع العميل صورة لقطعة مجوهرات.

وصف الصورة: {description}

نتائج البحث عن قطع مشابهة:
{search_results}

قم بتحليل الصورة وعرض القطع المشابهة من نتائج البحث. ركز على القطع الأكثر تشابهاً من ناحية التصميم والمواد والطراز."""

                # Get response without tool calling (since we already searched)
                response = openai.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "أنت مساعد مبيعات ودود في متجر مجوهرات. حلل الصورة واعرض المنتجات المشابهة بحماس."},
                        {"role": "user", "content": analysis_query}
                    ],
                    temperature=0.3
                )
                bot_response = response.choices[0].message.content

                # Display results
                st.markdown(bot_response)

                # Product cards disabled - all info in conversational text
                # if formatted_results:
                #     display_products(formatted_results)
                # else:
                #     st.info("لم يتم العثور على منتجات مشابهة")

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

            # Use function calling approach - ONE LLM call with full context and tools
            response = get_ai_response_with_tools(prompt, st.session_state.messages)

            thinking_placeholder.markdown(response)

        # Add to history after display
        st.session_state.messages.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ])

