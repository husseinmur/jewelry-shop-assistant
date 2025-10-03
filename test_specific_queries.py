#!/usr/bin/env python3
"""
Test specific queries that should bypass clarifying questions
"""

import sys
import os
sys.path.append('/home/hussein/shop-assistant')

from shared.config import init_apis
import openai
import json

def test_specific_queries():
    print("🎯 Testing Specific Queries (Should Bypass Clarification)")
    print("=" * 60)

    try:
        openai_client, pinecone_index = init_apis()

        # Test specific queries that should trigger direct search
        specific_queries = [
            "خاتم ذهب للزواج",
            "سلسلة بسيطة للاستعمال اليومي",
            "أقراط فضة أنيقة",
            "عقد ذهب فاخر للخطوبة",
            "هدية خاتم ذهب لزوجتي"
        ]

        for query in specific_queries:
            print(f"\n🔍 Testing: '{query}'")
            print("-" * 40)

            # Create test system (simplified)
            system_prompt = """أنت مساعد مبيعات ذكي وودود في متجر مجوهرات.

لديك أداتان:
- search_jewelry_products: للبحث عن منتجات محددة
- ask_clarifying_questions: لطرح أسئلة توضيحية عند الحاجة

🤔 متى تطرح أسئلة توضيحية:
- إذا قال العميل "مجوهرات" أو "شيء جميل" فقط
- إذا لم يحدد النوع أو المناسبة أو النمط

🔍 متى تبحث مباشرة:
- إذا حدد العميل النوع والمواد (مثل: خاتم ذهب)
- إذا ذكر المناسبة أو الاستخدام
- إذا كان الطلب واضح ومحدد"""

            # Define tools
            search_tool = {
                "type": "function",
                "function": {
                    "name": "search_jewelry_products",
                    "description": "Search for jewelry products",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"]
                    }
                }
            }

            ask_clarification_tool = {
                "type": "function",
                "function": {
                    "name": "ask_clarifying_questions",
                    "description": "Ask clarifying questions when request is vague",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {"type": "string"},
                            "questions": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["reason", "questions"]
                    }
                }
            }

            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                tools=[search_tool, ask_clarification_tool],
                tool_choice="auto",
                temperature=0.1
            )

            response_message = response.choices[0].message

            if response_message.tool_calls:
                tool_call = response_message.tool_calls[0]
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                if function_name == "search_jewelry_products":
                    search_query = function_args.get("query", "")
                    print(f"✅ Direct search triggered!")
                    print(f"   Search query: '{search_query}'")
                elif function_name == "ask_clarifying_questions":
                    print("❌ Clarification triggered (should have searched directly)")
                    reason = function_args.get("reason", "")
                    questions = function_args.get("questions", [])
                    print(f"   Reason: {reason}")
                    print(f"   Questions: {len(questions)}")
            else:
                print("💬 Direct response (no tools)")

        print(f"\n🎉 Testing completed!")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_specific_queries()