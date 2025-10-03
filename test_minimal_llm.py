#!/usr/bin/env python3
"""
Test minimal LLM prompt to find the issue
"""

import sys
sys.path.append('/home/hussein/shop-assistant')

from shared.config import init_apis

def test_minimal_llm():
    print("🧪 Testing Minimal LLM Response")
    print("=" * 40)

    try:
        openai_client, _ = init_apis()

        # Test 1: Very simple prompt
        print("Test 1: Simple prompt")
        response = openai_client.chat.completions.create(
            model="gpt-5-nano-2025-08-07",
            messages=[{"role": "user", "content": "Say hello"}],
            max_completion_tokens=50
        )
        result = response.choices[0].message.content.strip()
        print(f"Response: '{result}' (length: {len(result)})")

        # Test 2: JSON request
        print("\nTest 2: JSON request")
        response = openai_client.chat.completions.create(
            model="gpt-5-nano-2025-08-07",
            messages=[{"role": "user", "content": "Return a JSON list with numbers 1, 2, 3: [1, 2, 3]"}],
            max_completion_tokens=50
        )
        result = response.choices[0].message.content.strip()
        print(f"Response: '{result}' (length: {len(result)})")

        # Test 3: Arabic JSON request
        print("\nTest 3: Arabic with JSON")
        response = openai_client.chat.completions.create(
            model="gpt-5-nano-2025-08-07",
            messages=[{"role": "user", "content": "Find rings (خواتم) and return JSON list: ['id1', 'id2']"}],
            max_completion_tokens=100
        )
        result = response.choices[0].message.content.strip()
        print(f"Response: '{result}' (length: {len(result)})")

        # Test 4: Simplified verification
        print("\nTest 4: Simplified verification")
        test_prompt = """
Products:
1. ID: abc123, Category: خواتم
2. ID: def456, Category: عقود

Query: خاتم
Return JSON list of خواتم IDs: ["abc123"]
"""
        response = openai_client.chat.completions.create(
            model="gpt-5-nano-2025-08-07",
            messages=[{"role": "user", "content": test_prompt}],
            max_completion_tokens=100
        )
        result = response.choices[0].message.content.strip()
        print(f"Response: '{result}' (length: {len(result)})")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_minimal_llm()