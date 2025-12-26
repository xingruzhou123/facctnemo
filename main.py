import asyncio
from nemoguardrails import RailsConfig, LLMRails


async def main():
    # Load the guardrails configuration from the config directory
    config = RailsConfig.from_path("./config")

    # Create the LLMRails instance
    rails = LLMRails(config)

    print("NeMo Guardrails initialized with Qwen/Qwen3-0.6B via vLLM")
    print("=" * 50)

    # Example conversations
    test_messages = [
        "Hello!",
        "What can you do?",
        "Tell me about yourself.",
    ]

    for message in test_messages:
        print(f"\nUser: {message}")
        response = await rails.generate_async(messages=[{
            "role": "user",
            "content": message
        }])
        print(f"Bot: {response['content']}")

    print("\n" + "=" * 50)
    print("Interactive mode (type 'quit' to exit):")

    # Interactive conversation loop
    conversation_history = []
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break

        conversation_history.append({
            "role": "user",
            "content": user_input
        })

        response = await rails.generate_async(messages=conversation_history)
        print(f"Bot: {response['content']}")

        conversation_history.append({
            "role": "assistant",
            "content": response['content']
        })


if __name__ == "__main__":
    asyncio.run(main())
