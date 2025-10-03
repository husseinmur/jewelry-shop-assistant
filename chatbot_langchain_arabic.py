import streamlit as st
from PIL import Image
import json
from datetime import datetime
from shared.config import init_apis, TEXT_MODEL
# from shared.langchain_rag import init_langchain_rag  # No longer needed
from shared.embeddings import get_image_description
# from shared.database import search_by_image  # No longer needed - using optimized search
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

# RAG system no longer needed - using direct Pinecone + LLM verification

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "chat"



# Initialize APIs only (no RAG system needed)
try:
    openai_client, pinecone_index = init_apis()

    if not openai_client or not pinecone_index:
        st.error("❌ فشل في تهيئة نظام البحث")
        st.stop()

except Exception as e:
    st.error(f"خطأ في الاتصال: {e}")
    st.stop()

st.title("💎 مساعد متجر المجوهرات")

def llm_filter_results(query: str, results: list, openai_client) -> list:
    """Use LLM to intelligently filter search results for relevance"""
    try:
        if not results:
            return []

        # Format results for verification
        results_text = ""
        for i, result in enumerate(results):
            metadata = result.metadata
            results_text += f"ID: {result.id}\n"
            results_text += f"Name: {metadata.get('name', 'N/A')}\n"
            results_text += f"Category: {metadata.get('category', 'N/A')}\n"
            results_text += f"Description: {metadata.get('description', 'N/A')[:100]}...\n"
            results_text += f"Score: {result.score:.3f}\n\n"

        verification_prompt = f"""
Query: "{query}"

Available products to evaluate:
{results_text}

Task: Return product IDs that would satisfy the customer's search intent.

IMPORTANT CONTEXT & FLEXIBILITY RULES:
📿 JEWELRY USE CASES - Understand customer intent:
- "خاتم خطوبة" (engagement ring) = elegant rings with stones, gold, suitable for proposals
- "خاتم زواج" (wedding ring) = classic rings, bands, gold/platinum, suitable for marriage
- "خاتم بسيط" (simple ring) = minimal design, clean lines, not overly decorative
- "عقد فاخر" (luxury necklace) = high-end necklaces, precious materials, sophisticated design
- "أقراط يومية" (daily earrings) = comfortable, suitable for everyday wear
- "مجوهرات هدية" (gift jewelry) = presentable pieces, nice packaging appeal

🎯 MATCHING STRATEGY:
- Focus on SUITABILITY for the intended use, not exact terminology
- A beautiful gold ring with stones IS suitable for engagement even if not labeled "engagement ring"
- A simple gold band IS suitable for wedding even if not labeled "wedding ring"
- Consider material, design style, and appropriateness for the occasion

💎 MATERIAL & STYLE UNDERSTANDING:
- "ذهب" includes all gold types (yellow, white, rose gold)
- "بسيط" means clean, minimal, not overly decorative
- "فاخر" means luxury materials, sophisticated design, higher quality
- "أنيق" means elegant, refined, sophisticated

✅ EXAMPLES:
Query: "خاتم خطوبة" → Return: elegant rings with stones, gold rings suitable for proposals
Query: "سلسلة بسيطة" → Return: minimal necklaces, clean design chains
Query: "أقراط ذهب" → Return: any gold earrings regardless of specific style

❌ ONLY EXCLUDE if products are completely wrong category or material
- Query: "خاتم" (ring) → Don't return necklaces or earrings
- Query: "ذهب" (gold) → Don't return silver-only items

Return ONLY a JSON list of product IDs that match the customer's intent: ["id1", "id2", "id3"] or []
"""

        response = openai_client.chat.completions.create(
            model="gpt-5-nano-2025-08-07",
            messages=[{"role": "user", "content": verification_prompt}],
            temperature=1.0,
            max_completion_tokens=2000
        )

        # Parse response to get filtered IDs
        response_text = response.choices[0].message.content.strip()
        print(f"🐛 DEBUG LLM Response: '{response_text}' (length: {len(response_text)})")
        try:
            import json
            filtered_ids = json.loads(response_text)
            if not isinstance(filtered_ids, list):
                print(f"🐛 DEBUG - Response not a list: {type(filtered_ids)}")
                filtered_ids = []
            else:
                print(f"🐛 DEBUG - Parsed {len(filtered_ids)} IDs successfully")
        except Exception as e:
            print(f"🐛 DEBUG - JSON parsing failed: {e}")
            filtered_ids = []

        # Filter original results by verified IDs
        filtered_results = [r for r in results if r.id in filtered_ids]
        print(f"🐛 DEBUG LLM Filter - Input: {len(results)}, LLM IDs: {len(filtered_ids)}, Output: {len(filtered_results)}")
        return filtered_results

    except Exception as e:
        print(f"🐛 DEBUG LLM Filter - Exception: {e}")
        st.error(f"Error in LLM filtering: {e}")
        # Fallback to similarity filtering
        fallback_results = [r for r in results if r.score >= 0.4][:5]
        print(f"🐛 DEBUG LLM Filter - Using fallback: {len(fallback_results)} results")
        return fallback_results

def search_jewelry_products(query: str, conversation_history: list = None) -> str:
    """Search for jewelry products using direct Pinecone + LLM verification"""
    try:
        if not pinecone_index:
            return "نظام البحث غير متاح حالياً."

        # Get text embedding for the query
        from shared.embeddings import get_text_embedding
        query_embedding = get_text_embedding(query)

        if not query_embedding:
            return "فشل في معالجة الاستعلام."

        # Search Pinecone directly
        pinecone_results = pinecone_index.query(
            vector=query_embedding,
            top_k=8,  # Reduced from 15 to avoid overwhelming LLM
            include_metadata=True
        )

        # Basic similarity filter first (fast)
        decent_results = [r for r in pinecone_results.matches if r.score >= 0.3]

        if not decent_results:
            return "NO_RESULTS_NEED_CLARIFICATION"

        # LLM verification filter (intelligent)
        filtered_results = llm_filter_results(query, decent_results, openai_client)

        if not filtered_results:
            return "NO_RESULTS_NEED_CLARIFICATION"

        # Format results for LLM context
        final_results = filtered_results[:5]  # Top 5 verified results
        products_info = f"تم العثور على {len(final_results)} منتج مطابق في المخزون:\n\n"

        for i, result in enumerate(final_results, 1):
            metadata = result.metadata
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
            products_info += f"   الوصف: {metadata.get('description', '')[:150]}...\n"
            products_info += f"   مستوى التطابق: {result.score * 100:.1f}%\n\n"

        # Add instruction for LLM
        products_info += "\nتعليمات: تحدث بأسلوب دافئ ومرحب وودود. اذكر هذه المنتجات في إجابتك مع الأسعار والتفاصيل المهمة. تأكد من إدراج الرابط إذا كان متوفراً. تذكر: أنت تحافظ على سياق المحادثة وتربط إجابتك بما تم مناقشته سابقاً."

        return products_info

    except Exception as e:
        return f"حدث خطأ في البحث: {e}"

def ask_clarifying_questions(reason: str, questions: list) -> str:
    """Handle clarifying questions for vague queries"""
    try:
        response = f"أريد أن أساعدك في العثور على أفضل القطع! 😊\n\n"
        response += f"{reason}\n\n"
        response += "لذلك، هل يمكنك مساعدتي ببعض التفاصيل:\n\n"

        for i, question in enumerate(questions, 1):
            response += f"{i}. {question}\n"

        response += f"\nبهذه الطريقة سأتمكن من عرض أفضل القطع التي تناسب ذوقك تماماً! ✨"

        return response

    except Exception as e:
        return f"عذراً، حدث خطأ: {e}"

def get_ai_response_for_image_search(image_description: str, conversation_history: list) -> str:
    """Special function for image search that doesn't ask clarifying questions"""
    try:
        # Get OpenAI client
        openai_client, pinecone_index = init_apis()

        # Search for products based on image description
        search_result = search_jewelry_products(image_description, conversation_history)

        # If no results found, return appropriate message for image search
        if search_result == "NO_RESULTS_NEED_CLARIFICATION":
            return "لم أتمكن من العثور على قطع مشابهة للصورة التي رفعتها في مجموعتنا الحالية. 😔\n\nيمكنك تجربة:\n• رفع صورة أخرى أو بزاوية مختلفة\n• وصف القطعة التي تبحث عنها نصياً\n• تصفح مجموعتنا للعثور على قطع مشابهة 💎"

        # If results found, create response
        image_query = f"وصف الصورة: {image_description}\n\nنتائج البحث:\n{search_result}"

        # Create simple response for image analysis
        response = openai_client.chat.completions.create(
            model="gpt-5-nano-2025-08-07",
            messages=[
                {"role": "system", "content": "أنت مساعد مبيعات ودود في متجر مجوهرات. حلل الصورة المرفوعة واعرض المنتجات المشابهة بحماس. اذكر التشابه في التصميم أو المواد أو الطراز. كن ودود ومتحمس."},
                {"role": "user", "content": image_query}
            ],
            temperature=1.0
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"عذراً، حدث خطأ في تحليل الصورة: {e}"

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

        # Define the clarification tool
        ask_clarification_tool = {
            "type": "function",
            "function": {
                "name": "ask_clarifying_questions",
                "description": "Ask clarifying questions when the customer's request is too vague or lacks important details for a good search",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Why clarification is needed"
                        },
                        "questions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of clarifying questions to ask"
                        }
                    },
                    "required": ["reason", "questions"]
                }
            }
        }

        # Build conversation context summary FIRST
        context_summary = ""
        if conversation_history:
            recent_history = conversation_history[-4:] if len(conversation_history) > 4 else conversation_history
            if recent_history:
                context_summary = "📋 ملخص المحادثة السابقة:\n"
                for msg in recent_history:
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
                "content": f"""أنت مساعد مبيعات ذكي وودود في متجر مجوهرات.

{context_summary}

🔗 CRITICAL: إذا سأل العميل عن "السعر" أو "الألوان" أو "متوفر" بدون تحديد المنتج،
يجب أن تربط السؤال بآخر منتج ذكرته في المحادثة أعلاه.

قواعد أساسية:
1. راجع المحادثة السابقة أولاً قبل أي شيء
2. اربط الأسئلة الغامضة بالسياق السابق
3. لديك أداتان مهمتان:
   - search_jewelry_products: للبحث عن منتجات محددة
   - ask_clarifying_questions: لطرح أسئلة توضيحية عند الحاجة
4. لا تخترع معلومات - استخدم فقط ما في البحث أو المحادثة
5. كن ودوداً ومتحمساً

🤔 متى تطرح أسئلة توضيحية:

🚨 ASK CLARIFICATION for these vague cases:
- "jewelry" or "مجوهرات" ALONE (no type at all)
- "gift" or "هدية" ALONE (no details at all)
- "something nice" or "شيء جميل" (completely vague)
- "ring" or "خاتم" ALONE (type only, needs material/style/occasion)
- "necklace" or "عقد" ALONE (type only, needs material/style)
- "earrings" or "أقراط" ALONE (type only, needs material/style)

✅ SEARCH DIRECTLY - These have enough specificity:
✅ "gold ring" → SEARCH (type + material)
✅ "خاتم ذهب" → SEARCH (type + material)
✅ "silver earrings" → SEARCH (type + material)
✅ "أقراط فضة" → SEARCH (type + material)
✅ "simple necklace" → SEARCH (type + style)
✅ "عقد بسيط" → SEARCH (type + style)
✅ "wedding ring" → SEARCH (type + occasion)
✅ "خاتم زواج" → SEARCH (type + occasion)
✅ "engagement ring" → SEARCH (type + occasion)
✅ "خاتم خطوبة" → SEARCH (type + occasion)

❓ ASK CLARIFICATION:
❓ "ring" → ASK (needs material/style/occasion)
❓ "خاتم" → ASK (needs material/style/occasion)
❓ "necklace" → ASK (needs material/style)
❓ "عقد" → ASK (needs material/style)
❓ "jewelry" → ASK (no type specified)
❓ "مجوهرات" → ASK (no type specified)
❓ "gift" → ASK (no details at all)
❓ "هدية" → ASK (no details at all)

🎯 Key Rule: Customer needs TYPE + at least ONE additional detail (material/style/occasion) to search!

⚠️ CRITICAL: If customer says ONLY "ring", "خاتم", "necklace", "عقد", "earrings", or "أقراط" without any additional details, you MUST use ask_clarifying_questions tool!

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
            model="gpt-5-nano-2025-08-07",
            messages=messages,
            tools=[search_tool, ask_clarification_tool],
            tool_choice="auto",  # Let AI decide when to use tools
            temperature=1.0
        )

        response_message = response.choices[0].message

        # Check if AI wants to use tools
        if response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                function_args = json.loads(tool_call.function.arguments)

                if tool_call.function.name == "search_jewelry_products":
                    # Extract search query
                    search_query = function_args.get("query", "")

                    # Perform search
                    search_result = search_jewelry_products(search_query, conversation_history)

                    # Check if search failed and needs clarification
                    if search_result == "NO_RESULTS_NEED_CLARIFICATION":
                        # Automatically trigger clarification instead of showing failure
                        default_questions = [
                            "ما نوع المجوهرات التي تبحث عنها؟ (خاتم، عقد، أقراط، سوار)",
                            "ما المناسبة؟ (زواج، خطوبة، هدية، استعمال يومي)",
                            "ما المادة المفضلة؟ (ذهب، فضة، أحجار كريمة)",
                            "ما النمط المفضل؟ (بسيط، فاخر، عصري، كلاسيكي)"
                        ]
                        return ask_clarifying_questions(
                            "أريد أن أساعدك في العثور على القطعة المثالية! 💎",
                            default_questions
                        )

                    # Add tool result to conversation
                    messages.append(response_message)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": search_result
                    })

                    # Get final response with search results
                    final_response = openai.chat.completions.create(
                        model="gpt-5-nano-2025-08-07",
                        messages=messages,
                        temperature=1.0
                    )

                    return final_response.choices[0].message.content

                elif tool_call.function.name == "ask_clarifying_questions":
                    # Extract clarification parameters
                    reason = function_args.get("reason", "")
                    questions = function_args.get("questions", [])

                    # Generate clarification response
                    clarification_result = ask_clarifying_questions(reason, questions)

                    return clarification_result

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
                st.info(f"🔍 تحليل الصورة: {description[:100]}...")

                # Use specialized image search function
                bot_response = get_ai_response_for_image_search(description, st.session_state.messages)

                # Display results
                st.markdown(bot_response)

                # Add to chat history
                st.session_state.messages.append({
                    "role": "user",
                    "content": "🖼️ رفعت صورة قطعة مجوهرات"
                })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": bot_response
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

