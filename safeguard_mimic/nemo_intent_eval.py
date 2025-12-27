#!/usr/bin/env python3
"""
NeMo Guardrails Intent Classification Evaluation Script
Mimics SafeGuard's topical rails evaluation logic

This script evaluates intent classification using NeMo Guardrails with:
- Qwen as the main LLM for intent classification (via rails.generate_async)
- LlamaGuard for semantic safety checks (via action dispatcher)

Datasets: Banking77 + SmallTalk (JSON format with {"text": ..., "intent": ...})
"""

import os
import sys
import asyncio
import json
import csv
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("nemoguardrails").setLevel(logging.WARNING)

# Add script directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.embeddings.index import EmbeddingsIndex


# =============================================================================
# Configuration (matching safeguard_intent_config.py)
# =============================================================================

DATASET_CONFIGS = [
    {
        "name": "Banking77",
        "type": "JSON_LOCAL",
        "paths": {
            "ALL": os.path.join(SCRIPT_DIR, "datasets", "banking77_ALL.json"),
            "K3": os.path.join(SCRIPT_DIR, "datasets", "banking77_K3.json"),
            "K1": os.path.join(SCRIPT_DIR, "datasets", "banking77_K1.json"),
        },
    },
    {
        "name": "SmallTalk",
        "type": "JSON_LOCAL",
        "paths": {
            "ALL": os.path.join(SCRIPT_DIR, "datasets", "smalltalk_ALL.json"),
            "K3": os.path.join(SCRIPT_DIR, "datasets", "smalltalk_K3.json"),
            "K1": os.path.join(SCRIPT_DIR, "datasets", "smalltalk_K1.json"),
        },
    },
]

RESULT_DIR = os.path.join(SCRIPT_DIR, "eval_results")
os.makedirs(RESULT_DIR, exist_ok=True)

# Number of samples to use from each dataset file (set to 1 for quick testing)
NUM_SAMPLES_PER_FILE = 1


# =============================================================================
# Data Loader (matching safeguard_intent_data_loader.py)
# =============================================================================

def load_json_local(path: str) -> List[Dict]:
    """Reads a JSON file and returns its content."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset(cfg: Dict, max_samples: int = None) -> Dict[str, List[Dict]]:
    """
    Load dataset based on type.

    For JSON_LOCAL:
        return {
            "ALL": [...],
            "K3": [...],
            "K1": [...]
        }
    """
    dtype = cfg["type"]

    if dtype == "JSON_LOCAL":
        result = {}
        for split_name, path in cfg["paths"].items():
            if not os.path.exists(path):
                log.warning(f"Dataset file not found: {path}")
                result[split_name] = []
                continue

            data = load_json_local(path)

            if not isinstance(data, list):
                raise ValueError(f"JSON file must contain a list, got: {type(data)}")

            # Limit samples if specified
            if max_samples and max_samples > 0:
                data = data[:max_samples]

            result[split_name] = data
        return result
    else:
        raise ValueError(f"Unsupported dataset type: {dtype}")


# =============================================================================
# Intent Classification Prompt
# =============================================================================

INTENT_CLASSIFICATION_PROMPT = """You are an intent classifier for a banking/customer service application.
Given the user query, classify it into one of the intent categories.

Common Banking Intents:
- card_arrival, card_linking, exchange_rate, card_payment_wrong_exchange_rate
- extra_charge_on_statement, pending_cash_withdrawal, fiat_currency_support
- card_delivery_estimate, automatic_top_up, card_not_working, exchange_via_app
- lost_or_stolen_card, age_limit, pin_blocked, contactless_not_working
- top_up_by_bank_transfer_charge, pending_top_up, cancel_transfer, top_up_limits
- card_payment_fee_charged, transfer_not_received_by_recipient, supported_cards_and_currencies
- getting_virtual_card, card_acceptance, top_up_reverted, balance_not_updated_after_cheque_or_cash_deposit
- card_payment_not_recognised, edit_personal_details, why_verify_identity, unable_to_verify_identity
- get_physical_card, visa_or_mastercard, topping_up_by_card, disposable_card_limits
- compromised_card, atm_support, direct_debit_payment_not_recognised, passcode_forgotten
- declined_cash_withdrawal, pending_card_payment, lost_or_stolen_phone, request_refund
- declined_transfer, Refund_not_showing_up, declined_card_payment, pending_transfer
- terminate_account, card_swallowed, transaction_charged_twice, verify_source_of_funds
- transfer_timing, reverted_card_payment?, change_pin, beneficiary_not_allowed
- transfer_fee_charged, receiving_money, failed_transfer, transfer_into_account
- verify_top_up, getting_spare_card, top_up_by_cash_or_cheque, order_physical_card
- virtual_card_not_working, wrong_exchange_rate_for_cash_withdrawal, get_disposable_virtual_card
- top_up_failed, balance_not_updated_after_bank_transfer, cash_withdrawal_not_recognised
- exchange_charge, top_up_by_card_charge, activate_my_card, cash_withdrawal_charge
- card_about_to_expire, apple_pay_or_google_pay, verify_my_identity, country_support
- wrong_amount_of_cash_received

Common SmallTalk Intents:
- smalltalk_agent_acquaintance, smalltalk_agent_age, smalltalk_agent_annoying
- smalltalk_agent_answer_my_question, smalltalk_agent_bad, smalltalk_agent_be_clever
- smalltalk_agent_beautiful, smalltalk_agent_birth_date, smalltalk_agent_boring
- smalltalk_agent_boss, smalltalk_agent_busy, smalltalk_agent_chatbot
- smalltalk_greetings_hello, smalltalk_greetings_bye, smalltalk_appraisal_thank_you

User query: {query}

Output ONLY the intent label, nothing else:"""


# =============================================================================
# LLM Interface using NeMo Guardrails
# =============================================================================

class NemoLLMInterface:
    """
    Encapsulates NeMo Guardrails for intent classification evaluation.
    Uses Qwen as main LLM via rails.generate_async().
    Uses LlamaGuard for safety checks via action dispatcher.
    """

    def __init__(self, rails: LLMRails):
        self.rails = rails
        self.action_dispatcher = rails.runtime.action_dispatcher
        log.info("[NemoLLMInterface] Initialized with NeMo Guardrails")

    async def classify_intent(self, text: str) -> str:
        """
        Classify user intent using NeMo Guardrails LLM (Qwen).
        Calls the LLM directly to bypass output rails (RAG check interference).

        Returns:
            Predicted intent string
        """
        import re
        import httpx

        if not text:
            return "none"

        try:
            # Build the intent classification prompt
            prompt = INTENT_CLASSIFICATION_PROMPT.format(query=text)

            log.info(f"[NemoLLMInterface] Calling LLM for: '{text[:50]}...'")

            # Get LLM config from NeMo rails
            main_model = None
            for m in self.rails.config.models:
                if m.type == "main":
                    main_model = m
                    break

            if not main_model:
                log.error("No main model found in config")
                return "error"

            base_url = main_model.parameters.get("base_url", "http://localhost:8000/v1")
            model_name = main_model.model

            # Call Qwen directly (bypass output rails to avoid RAG check interference)
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{base_url}/chat/completions",
                    json={
                        "model": model_name,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": 500,  # Increased to allow thinking + intent output
                        "temperature": 0.0,
                        # vLLM passes chat_template_kwargs directly at top level
                        "chat_template_kwargs": {
                            "enable_thinking": False  # Disable thinking mode
                        }
                    },
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                result = response.json()
                llm_response = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

            log.info(f"[NemoLLMInterface] LLM raw response: '{llm_response[:100]}...'")

            # Clean thinking tags if present (Qwen sometimes uses <think>...</think>)
            clean_response = re.sub(r'<think>.*?</think>', '', llm_response, flags=re.DOTALL | re.IGNORECASE)
            clean_response = clean_response.strip()

            # Extract just the intent (first line, no extra text)
            intent = clean_response.split('\n')[0].strip()

            # Remove any quotes or extra characters
            intent = intent.strip('"\'`')

            log.info(f"[NemoLLMInterface] Intent classification: '{text[:40]}...' -> '{intent}'")
            return intent

        except Exception as e:
            log.error(f"[NemoLLMInterface] Error during classification: {e}")
            import traceback
            traceback.print_exc()
            return "error"

    async def check_safety(self, text: str) -> str:
        """
        Check if text is safe using LlamaGuard through NeMo action dispatcher.

        Returns:
            "safe" or "unsafe"
        """
        try:
            # Call LlamaGuard through NeMo's action dispatcher
            result = await self.action_dispatcher.execute_action(
                "call_llama_guard_input",
                {"text": text}
            )
            # Action returns True if UNSAFE
            is_unsafe = result[0] if isinstance(result, tuple) else result
            safety_status = "unsafe" if is_unsafe else "safe"
            log.info(f"[NemoLLMInterface] Safety check: '{text[:40]}...' -> {safety_status}")
            return safety_status
        except Exception as e:
            log.warning(f"[NemoLLMInterface] Safety check failed: {e}")
            return "safe"  # Default to safe on error

    async def process_sample(self, text: str) -> Dict[str, str]:
        """
        Process a single sample: classify intent and check safety.
        """
        intent = await self.classify_intent(text)
        safety = await self.check_safety(text)
        return {
            "predicted_intent": intent,
            "safety_check": safety
        }


# =============================================================================
# Evaluator (matching safeguard_intent_evaluator.py)
# =============================================================================

class Evaluator:
    """
    Evaluate Intent classification via NeMo Guardrails pipeline.
    """

    def __init__(self, llm_interface: NemoLLMInterface):
        self.llm_interface = llm_interface

    async def evaluate_dataset_async(self, dataset: List[Dict[str, str]], out_csv: str) -> float:
        """
        Evaluate intent classification asynchronously.

        Args:
            dataset: list[{"text": ..., "intent": ...}]
            out_csv: path to output the results

        Returns:
            float: intent classification accuracy.
        """
        if not dataset:
            log.warning("Empty dataset, returning 0.0 accuracy")
            return 0.0

        log.info(f"[Evaluator] Running intent prediction on {len(dataset)} samples...")

        # Process each sample
        correct_count = 0
        rows = []

        for idx, sample in enumerate(dataset):
            text = sample["text"]
            expected_intent = sample["intent"]

            log.info(f"[Evaluator] Processing sample {idx + 1}/{len(dataset)}: '{text[:50]}...'")

            try:
                result = await self.llm_interface.process_sample(text)
                predicted_intent = result["predicted_intent"]
                safety = result["safety_check"]
            except Exception as e:
                log.error(f"[Evaluator] Error processing sample: {e}")
                predicted_intent = "error"
                safety = "error"

            # Compare intents (case-insensitive)
            pred_intent_str = str(predicted_intent).lower().strip()
            exp_intent_str = str(expected_intent).lower().strip()

            is_correct = exp_intent_str == pred_intent_str
            if is_correct:
                correct_count += 1

            rows.append({
                "text": text,
                "expected_intent": expected_intent,
                "predicted_intent": predicted_intent,
                "is_correct": is_correct,
                "safety_check": safety,
            })

            # Log result
            status = "CORRECT" if is_correct else "WRONG"
            log.info(f"[Evaluator] [{status}] Expected: '{exp_intent_str}', Got: '{pred_intent_str}'")

        # Calculate accuracy
        accuracy = correct_count / len(dataset) if dataset else 0.0

        log.info(f"[Evaluator] Accuracy: {accuracy:.4f} ({correct_count}/{len(dataset)})")

        # Write to CSV
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        log.info(f"[Evaluator] Saved detailed results to: {out_csv}")

        return accuracy

    async def evaluate_dataset(self, dataset: List[Dict[str, str]], out_csv: str) -> float:
        """
        Wrapper that calls evaluate_dataset_async.
        """
        return await self.evaluate_dataset_async(dataset, out_csv)


# =============================================================================
# Main Evaluation Runner (matching safeguard_intent_run_evaluation.py)
# =============================================================================

async def run_for_dataset(
    dataset_name: str,
    dataset_splits: Dict[str, List[Dict[str, str]]],
    evaluator: Evaluator,
) -> List[Dict]:
    """
    Perform evaluation on different K-shot splits.
    """
    summary_rows = []

    for k_name, samples in dataset_splits.items():
        if not samples:
            log.warning(f"Skipping {dataset_name} Split={k_name}: No samples found.")
            continue

        out_csv = os.path.join(RESULT_DIR, f"intent_{dataset_name}_{k_name}.csv")
        print(f"\n=== Running Evaluation: {dataset_name} Split={k_name} ({len(samples)} samples) ===")

        accuracy = await evaluator.evaluate_dataset(samples, out_csv)

        summary_rows.append({
            "dataset": dataset_name,
            "split": k_name,
            "num_samples": len(samples),
            "accuracy": f"{accuracy:.4f}",
            "output_file": out_csv,
        })

    return summary_rows


async def init_nemo_guardrails() -> LLMRails:
    """Initialize NeMo Guardrails with dummy embeddings."""

    # Dummy embeddings index to avoid downloads
    class DummyIndex(EmbeddingsIndex):
        @property
        def embedding_size(self):
            return 0

        @property
        def cache_config(self):
            return None

        async def _get_embeddings(self, texts):
            return []

        async def add_item(self, item):
            return None

        async def add_items(self, items):
            return None

        async def build(self):
            return None

        async def search(self, text, max_results, threshold=None):
            return []

    LLMRails._get_embeddings_search_provider_instance = (
        lambda self, esp=None: DummyIndex()
    )

    config = RailsConfig.from_path(SCRIPT_DIR)
    rails = LLMRails(config)

    return rails


async def main_async():
    """Main async evaluation function."""
    print("=" * 70)
    print("NeMo Guardrails Intent Classification Evaluation")
    print("(Using Qwen via rails.generate_async + LlamaGuard via action dispatcher)")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Samples per file: {NUM_SAMPLES_PER_FILE}")
    print()

    # Initialize NeMo Guardrails
    print("Initializing NeMo Guardrails...")
    try:
        rails = await init_nemo_guardrails()
        print(f"  Colang version: {rails.config.colang_version}")
        print(f"  Models configured: {[m.type for m in rails.config.models]}")
        for m in rails.config.models:
            print(f"    - {m.type}: {m.model}")
    except Exception as e:
        print(f"ERROR: Failed to initialize NeMo Guardrails: {e}")
        import traceback
        traceback.print_exc()
        return

    # Verify actions are registered
    action_dispatcher = rails.runtime.action_dispatcher
    registered_actions = list(action_dispatcher.registered_actions.keys())
    print(f"  Registered actions: {len(registered_actions)}")

    if "call_llama_guard_input" in registered_actions:
        print("  [OK] call_llama_guard_input action registered")
    else:
        print(f"  [WARN] call_llama_guard_input not found")
    print()

    # Create interface and evaluator
    llm_interface = NemoLLMInterface(rails)
    evaluator = Evaluator(llm_interface)

    # Run evaluation
    all_summary = []

    for cfg in DATASET_CONFIGS:
        print(f"\n--- Loading dataset: {cfg['name']} ---")
        try:
            dataset_splits = load_dataset(cfg, max_samples=NUM_SAMPLES_PER_FILE)

            # Log what we loaded
            for split_name, samples in dataset_splits.items():
                print(f"  {split_name}: {len(samples)} samples")

        except Exception as e:
            print(f"Error loading dataset {cfg['name']}: {e}. Skipping.")
            import traceback
            traceback.print_exc()
            continue

        summary = await run_for_dataset(cfg["name"], dataset_splits, evaluator)
        all_summary.extend(summary)

    # Save summary
    if all_summary:
        import pandas as pd
        df = pd.DataFrame(all_summary)
        summary_file = os.path.join(RESULT_DIR, "intent_summary.csv")
        df.to_csv(summary_file, index=False)

        print("\n" + "=" * 70)
        print("      Intent Classification Evaluation Summary")
        print("=" * 70)
        print(df.to_string(index=False))
        print("-" * 70)
        print(f"Summary saved to: {summary_file}")
    else:
        print("\n=== Evaluation Failed or No Data Processed ===")

    print("\nDone!")


def main():
    """Entry point."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
