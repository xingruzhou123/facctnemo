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

# Set up file handler for all logs
file_handler = logging.FileHandler(log_file, mode='w')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

# Configure root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(file_handler)

# Configure nemoguardrails logger specifically
nemo_logger = logging.getLogger("nemoguardrails")
nemo_logger.setLevel(logging.DEBUG)

# Keep console quiet (only warnings+)
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
root_logger.addHandler(console)

# Our app logger
logger = logging.getLogger(__name__)


async def main():
    # Load the guardrails configuration from the config directory
    config = RailsConfig.from_path("./config")

    # Create the LLMRails instance
    rails = LLMRails(config)

    print("NeMo Guardrails initialized with Qwen/Qwen3-0.6B via vLLM")
    print(f"Logs: {log_file}")
    print("=" * 50)
    logger.info("=== Session started ===")

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
        bot_reply = response.response[0]['content']
        print(f"Bot: {bot_reply}")

        # Log activated rails and blocking reasons to file
        logger.info(f"User message: {message}")
        logger.info(f"Bot response: {bot_reply}")
        if response.log:
            logger.info(f"Activated rails count: {len(response.log.activated_rails) if response.log.activated_rails else 0}")
            if response.log.activated_rails:
                for rail in response.log.activated_rails:
                    logger.info(f"Rail: {rail.type} ({rail.name}) - stop={rail.stop}")
                    logger.info(f"  Decisions: {rail.decisions}")
                    if rail.executed_actions:
                        logger.info(f"  Actions: {[a.action_name for a in rail.executed_actions]}")
        else:
            logger.info("No response.log available")

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

        # Log activated rails and blocking reasons to file
        logger.info(f"User message: {user_input}")
        logger.info(f"Bot response: {bot_reply}")
        if response.log:
            logger.info(f"Activated rails count: {len(response.log.activated_rails) if response.log.activated_rails else 0}")
            if response.log.activated_rails:
                for rail in response.log.activated_rails:
                    logger.info(f"Rail: {rail.type} ({rail.name}) - stop={rail.stop}")
                    logger.info(f"  Decisions: {rail.decisions}")
                    if rail.executed_actions:
                        logger.info(f"  Actions: {[a.action_name for a in rail.executed_actions]}")
        else:
            logger.info("No response.log available")

        conversation_history.append({
            "role": "assistant",
            "content": bot_reply
        })


if __name__ == "__main__":
    asyncio.run(main())
