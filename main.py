import asyncio
import logging
from datetime import datetime
from pathlib import Path

from nemoguardrails import RailsConfig, LLMRails

# Create logs folder
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Create unique log file for each run
log_file = log_dir / f"guardrails_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=log_file,
    filemode='w'
)

# Keep console quiet
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
logging.getLogger().addHandler(console)


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
        response = await rails.generate_async(
            messages=[{"role": "user", "content": message}],
            options={"log": {"activated_rails": True, "llm_calls": True}}
        )
        # Extract content from response
        # response.response is a list like [{'role': 'assistant', 'content': '...'}]
        bot_reply = response.response[0]['content']
        print(f"Bot: {bot_reply}")

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

        response = await rails.generate_async(
            messages=conversation_history,
            options={"log": {"activated_rails": True, "llm_calls": True}}
        )

        bot_reply = response.response[0]['content']
        print(f"Bot: {bot_reply}")

        conversation_history.append({
            "role": "assistant",
            "content": bot_reply
        })


if __name__ == "__main__":
    asyncio.run(main())
