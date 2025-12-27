#!/usr/bin/env python3
"""
NeMo Guardrails RAG Faithfulness Evaluation Script
Mimics SafeGuard's RAG evaluation logic

This script properly goes through NeMo Guardrails and uses LlamaGuard
for RAG faithfulness checking.

Tests on MS-MARCO style data:
- Positive samples: Answers that ARE grounded in the evidence (is_faithful=True)
- Negative samples: Answers that are NOT grounded in the evidence (is_faithful=False)

Metrics computed (matching SafeGuard):
- TP (True Positives): Faithful answers correctly identified as faithful
- TN (True Negatives): Unfaithful answers correctly identified as unfaithful
- FP (False Positives): Unfaithful answers incorrectly identified as faithful
- FN (False Negatives): Faithful answers incorrectly identified as unfaithful
"""

import os
import sys
import asyncio
import json
import logging
from typing import List, Dict
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

# Add safeguard_mimic to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from nemoguardrails import RailsConfig, LLMRails
from nemoguardrails.embeddings.index import EmbeddingsIndex

# Dataset path
FAITHFULNESS_DATA_PATH = os.path.join(SCRIPT_DIR, "datasets", "faithfulness_data_msmarco.jsonl")

# Number of samples to evaluate
NUM_SAMPLES = 10


def load_faithfulness_data(file_path: str, max_samples: int) -> List[Dict]:
    """Loads MS-MARCO style data from a JSONL file."""
    if not os.path.exists(file_path):
        print(f"ERROR: Dataset file not found: {file_path}")
        return []

    try:
        all_data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_data.append(json.loads(line.strip()))

        # Balance the samples: get equal positive and negative
        positive_samples = [d for d in all_data if d.get("is_faithful") is True]
        negative_samples = [d for d in all_data if d.get("is_faithful") is False]

        half_samples = max_samples // 2
        selected_pos = positive_samples[:half_samples]
        selected_neg = negative_samples[:half_samples]

        return selected_pos + selected_neg

    except json.JSONDecodeError as e:
        print(f"ERROR: JSON Decode Error: {e}")
        return []


async def evaluate_with_nemo_guardrails(rails: LLMRails, sample: Dict) -> Dict:
    """
    Evaluate a single sample using NeMo Guardrails with both LlamaGuard and Qwen.

    Flow through NeMo Guardrails:
    1. LlamaGuard: Safety check on the answer (call_llama_guard_output)
    2. Qwen: Faithfulness check (check_rag_faithfulness_llama - uses Qwen for NLI)

    Both checks go through NeMo Guardrails' action dispatcher.
    For RAG evaluation, we focus on faithfulness (Qwen) but also log LlamaGuard safety.
    """
    ground_truth = 1 if sample.get("is_faithful") is True else 0

    action_dispatcher = rails.runtime.action_dispatcher

    try:
        # Step 1: LlamaGuard safety check through NeMo
        # call_llama_guard_output returns True if UNSAFE, False if SAFE
        is_safe = True  # Default: safe
        try:
            llama_result = await action_dispatcher.execute_action(
                "call_llama_guard_output",
                {"text": sample["answer"]}
            )
            # Action dispatcher returns (result, status) tuple
            llama_guard_unsafe = llama_result[0] if isinstance(llama_result, tuple) else llama_result
            is_safe = not llama_guard_unsafe
            log.info(f"LlamaGuard: '{sample['answer'][:40]}...' -> unsafe={llama_guard_unsafe}, safe={is_safe}")
        except Exception as e:
            log.warning(f"LlamaGuard check failed: {e}")

        # Step 2: Qwen faithfulness check through NeMo
        # check_rag_faithfulness_llama returns True if FAITHFUL, False if UNFAITHFUL
        faithful_result = await action_dispatcher.execute_action(
            "check_rag_faithfulness_llama",
            {
                "response": sample["answer"],
                "context": sample["evidence"]
            }
        )
        # Action dispatcher returns (result, status) tuple
        is_faithful = faithful_result[0] if isinstance(faithful_result, tuple) else faithful_result

        # For RAG faithfulness evaluation, focus on faithfulness
        # Safety is logged but faithfulness determines the prediction
        prediction = 1 if is_faithful else 0

        log.info(f"Final: safe={is_safe}, faithful={is_faithful}, prediction={prediction}")

    except Exception as e:
        log.error(f"Error calling action through NeMo: {e}")
        prediction = 1  # Default on error

    return {
        "ground_truth": ground_truth,
        "prediction": prediction,
        "question": sample["question"][:60],
        "answer": sample["answer"][:60],
    }


async def test_full_guardrails_flow(rails: LLMRails, sample: Dict) -> Dict:
    """
    Alternative: Test through full NeMo Guardrails pipeline.

    This sends a question through the full pipeline:
    1. Input rails (LlamaGuard input check)
    2. LLM generates response
    3. Output rails (LlamaGuard output check + RAG faithfulness check)

    We pass the evidence as retrieved_docs in the context.
    """
    ground_truth = 1 if sample.get("is_faithful") is True else 0

    # Create message with context containing the evidence
    messages = [
        {"role": "context", "content": {"retrieved_docs": [sample["evidence"]]}},
        {"role": "user", "content": f"Based on the provided context, answer: {sample['question']}"}
    ]

    try:
        # Generate through full NeMo Guardrails pipeline
        result = await rails.generate_async(messages=messages)
        response = result.get("content", "") if isinstance(result, dict) else str(result)

        # Check if the response was blocked (indicates RAG check failed or safety issue)
        blocked_indicators = [
            "cannot verify",
            "cannot assist",
            "cannot provide",
            "safety guidelines",
            "refuse",
            "apologize"
        ]

        is_blocked = any(indicator in response.lower() for indicator in blocked_indicators)
        prediction = 0 if is_blocked else 1

        log.info(f"Full flow: Q='{sample['question'][:40]}...' -> blocked={is_blocked}")

    except Exception as e:
        log.error(f"Error in full guardrails flow: {e}")
        prediction = 1  # Default to faithful on error

    return {
        "ground_truth": ground_truth,
        "prediction": prediction,
        "question": sample["question"][:60],
        "answer": sample["answer"][:60],
    }


async def run_evaluation():
    """Main evaluation function."""
    print("=" * 60)
    print("NeMo Guardrails RAG Faithfulness Evaluation")
    print("(Using LlamaGuard through NeMo Guardrails)")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # 1. Load Data
    print("Loading dataset...")
    test_cases = load_faithfulness_data(FAITHFULNESS_DATA_PATH, NUM_SAMPLES)

    if not test_cases:
        print("ERROR: Could not load dataset. Aborting.")
        return

    total_pos = sum(1 for case in test_cases if case.get("is_faithful") is True)
    total_neg = sum(1 for case in test_cases if case.get("is_faithful") is False)

    print(f"  Loaded {len(test_cases)} samples")
    print(f"  - Positive (faithful): {total_pos}")
    print(f"  - Negative (unfaithful): {total_neg}")
    print()

    # 2. Initialize NeMo Guardrails
    print("Initializing NeMo Guardrails...")

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

    try:
        config = RailsConfig.from_path(SCRIPT_DIR)
        rails = LLMRails(config)
        print(f"  Colang version: {config.colang_version}")
        print(f"  Models: {[m.type for m in config.models]}")
        if config.rails and config.rails.output:
            print(f"  Output rails: {config.rails.output.flows}")
        print()
    except Exception as e:
        print(f"ERROR: Failed to initialize NeMo Guardrails: {e}")
        return

    # 3. Verify LlamaGuard action is registered
    print("Verifying LlamaGuard RAG action is registered...")
    action_dispatcher = rails.runtime.action_dispatcher
    registered_actions = list(action_dispatcher.registered_actions.keys())

    if "check_rag_faithfulness_llama" in registered_actions:
        print("  [OK] check_rag_faithfulness_llama action registered")
    else:
        print(f"  [WARN] check_rag_faithfulness_llama not found in: {registered_actions[:10]}...")
    print()

    # 4. Run Faithfulness Checks through NeMo Guardrails + LlamaGuard
    print("=" * 60)
    print("Running RAG Faithfulness Checks (NeMo + LlamaGuard)")
    print("=" * 60)

    TP = TN = FP = FN = 0

    for idx, case in enumerate(test_cases):
        print(f"  Processing {idx + 1}/{len(test_cases)}...", flush=True)

        # Use NeMo Guardrails action dispatcher to call LlamaGuard
        result = await evaluate_with_nemo_guardrails(rails, case)

        gt = result["ground_truth"]
        pred = result["prediction"]

        # Determine outcome
        if gt == 1 and pred == 1:
            TP += 1
            outcome = "TP"
        elif gt == 0 and pred == 0:
            TN += 1
            outcome = "TN"
        elif gt == 0 and pred == 1:
            FP += 1
            outcome = "FP"
        elif gt == 1 and pred == 0:
            FN += 1
            outcome = "FN"

        gt_label = "FAITHFUL" if gt == 1 else "UNFAITHFUL"
        pred_label = "FAITHFUL" if pred == 1 else "UNFAITHFUL"
        print(f"    [{outcome}] GT={gt_label}, Pred={pred_label}")
        print(f"         Q: {result['question']}...")
        print(f"         A: {result['answer']}...")

    print()

    # 5. Calculate Metrics
    total_checked = total_pos + total_neg
    overall_accuracy = ((TP + TN) / total_checked) if total_checked else 0.0
    pos_entailment_acc = (TP / total_pos) if total_pos else 0.0
    neg_entailment_acc = (TN / total_neg) if total_neg else 0.0
    precision = (TP / (TP + FP)) if (TP + FP) > 0 else 0.0
    recall = (TP / (TP + FN)) if (TP + FN) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # 6. Output Summary
    print("=" * 60)
    print("      RAG Faithfulness Performance Summary")
    print("      (NeMo Guardrails + LlamaGuard)")
    print("=" * 60)
    print(f"Total Samples: {total_checked} (Pos: {total_pos}, Neg: {total_neg})")
    print("-" * 60)
    print("Confusion Matrix:")
    print(f"  TP (Faithful correctly identified):   {TP:4d}")
    print(f"  TN (Unfaithful correctly identified): {TN:4d}")
    print(f"  FP (Unfaithful leaked as faithful):   {FP:4d}")
    print(f"  FN (Faithful blocked as unfaithful):  {FN:4d}")
    print("-" * 60)
    print("SafeGuard Comparison Metrics:")
    print(f"  Overall Accuracy:             {overall_accuracy:.4f}")
    print(f"  Positive Entailment Accuracy: {pos_entailment_acc:.4f}")
    print(f"  Negative Entailment Accuracy: {neg_entailment_acc:.4f}")
    print("-" * 60)
    print("Additional Metrics:")
    print(f"  Precision:                    {precision:.4f}")
    print(f"  Recall:                       {recall:.4f}")
    print(f"  F1 Score:                     {f1:.4f}")
    print("=" * 60)

    # 7. Save results
    output_dir = os.path.join(SCRIPT_DIR, "eval_results")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"rag_metrics_{timestamp}.txt")

    with open(results_file, "w") as f:
        f.write("NeMo Guardrails RAG Faithfulness Evaluation Results\n")
        f.write("(Using LlamaGuard through NeMo Guardrails)\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Total Samples: {total_checked}\n")
        f.write(f"Positive: {total_pos}, Negative: {total_neg}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(f"  TP: {TP}, TN: {TN}\n")
        f.write(f"  FP: {FP}, FN: {FN}\n")
        f.write("\nMetrics:\n")
        f.write(f"  Overall Accuracy: {overall_accuracy:.4f}\n")
        f.write(f"  Positive Entailment Accuracy: {pos_entailment_acc:.4f}\n")
        f.write(f"  Negative Entailment Accuracy: {neg_entailment_acc:.4f}\n")
        f.write(f"  Precision: {precision:.4f}\n")
        f.write(f"  Recall: {recall:.4f}\n")
        f.write(f"  F1 Score: {f1:.4f}\n")

    print(f"\nResults saved to: {results_file}")
    print("Done!")


if __name__ == "__main__":
    asyncio.run(run_evaluation())
