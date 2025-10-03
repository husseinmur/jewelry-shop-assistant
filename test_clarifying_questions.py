#!/usr/bin/env python3
"""
Test the clarifying questions feature
"""

import sys
import os
sys.path.append('/home/hussein/shop-assistant')

import time
from shared.config import init_apis
import openai
import json

def test_clarifying_questions():
    print("🤔 Testing Clarifying Questions Feature")
    print("=" * 50)

    try:
        # Initialize
        openai_client, pinecone_index = init_apis()
        print("✅ APIs initialized successfully")

        # Test queries that should trigger clarifying questions
        test_queries = [
            "مجوهرات",
            "شيء جميل",
            "هدية",
            "خاتم",  # More specific but could still ask for clarification
            "أريد أشتري مجوهرات"
        ]

        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            print("-" * 30)

            # Define tools (copy from main chatbot)
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

            # Create enhanced system prompt
            system_prompt = """أنت مساعد مبيعات ذكي وودود في متجر مجوهرات.

قواعد أساسية:
1. لديك أداتان مهمتان:
   - search_jewelry_products: للبحث عن منتجات محددة
   - ask_clarifying_questions: لطرح أسئلة توضيحية عند الحاجة
2. لا تخترع معلومات - استخدم فقط ما في البحث أو المحادثة
3. كن ودوداً ومتحمساً

🤔 متى تطرح أسئلة توضيحية:
- إذا قال العميل "مجوهرات" أو "شيء جميل" فقط
- إذا لم يحدد النوع: خاتم، عقد، أقراط، سوار
- إذا لم يذكر المناسبة: زواج، خطوبة، هدية، يومي
- إذا لم يحدد الميزانية أو النمط المفضل
- إذا طلب "هدية" بدون تحديد لمن أو ما المناسبة

🎯 أمثلة للأسئلة التوضيحية:
- ما نوع المجوهرات التي تفضل؟ (خاتم، عقد، أقراط، سوار)
- ما المناسبة؟ (زواج، خطوبة، هدية، استعمال يومي)
- ما الميزانية المناسبة لك؟
- ما النمط المفضل؟ (بسيط، فاخر، عصري، كلاسيكي)
- الهدية لمن؟ (زوجة، أم، أخت، صديقة)"""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]

            start_time = time.time()

            # Call OpenAI
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                tools=[search_tool, ask_clarification_tool],
                tool_choice="auto",
                temperature=0.3
            )

            response_time = time.time() - start_time
            response_message = response.choices[0].message

            print(f"Response time: {response_time:.2f}s")

            # Check what tool was called
            if response_message.tool_calls:
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    print(f"🔧 Tool called: {function_name}")

                    if function_name == "ask_clarifying_questions":
                        print("✅ Clarifying questions triggered!")
                        reason = function_args.get("reason", "")
                        questions = function_args.get("questions", [])

                        print(f"Reason: {reason}")
                        print("Questions:")
                        for i, q in enumerate(questions, 1):
                            print(f"  {i}. {q}")

                        # Format the response
                        clarification_response = f"أريد أن أساعدك في العثور على أفضل القطع! 😊\n\n"
                        clarification_response += f"{reason}\n\n"
                        clarification_response += "لذلك، هل يمكنك مساعدتي ببعض التفاصيل:\n\n"

                        for i, question in enumerate(questions, 1):
                            clarification_response += f"{i}. {question}\n"

                        clarification_response += f"\nبهذه الطريقة سأتمكن من عرض أفضل القطع التي تناسب ذوقك تماماً! ✨"

                        print(f"\n💬 Final response:\n{clarification_response}")

                    elif function_name == "search_jewelry_products":
                        search_query = function_args.get("query", "")
                        print(f"🔍 Search triggered with query: '{search_query}'")
                        print("⚠️  This query was specific enough to skip clarification")

            else:
                print("💬 Direct response (no tools):")
                print(response_message.content)

            print("\n" + "="*50)

        print("\n🎉 Testing completed!")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_clarifying_questions()