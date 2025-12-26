# Custom Actions for NeMo Guardrails
# Regex-based content filtering (mimicking SafeGuard Rails)

import re
import logging
from typing import Optional
from nemoguardrails.actions import action

log = logging.getLogger(__name__)

# =============================================================================
# Regex Patterns (from SafeGuard config)
# =============================================================================

# AMD sensitive information pattern
AMD_SENSITIVE_PATTERN = re.compile(
    r"(?i)(AMD|Ryzen|Zen\s*\d+).*(branch\s*predict(or|ion)|cache|micro[- ]?arch|pipeline|BTB|PBT|prediction\s*table)",
    re.IGNORECASE
)

# Harmful content keywords (input)
HARMFUL_INPUT_PATTERN = re.compile(
    r"(?i)(bomb|grenade|explosive|weapon|kill|murder|arson|accelerant|meth|"
    r"drug\s*(recipe|synthesis)|dark\s*web|hack|phishing|backdoor|voyeur|"
    r"secretly\s*(watch|record)|nude|slit\s*eyes|ballet\s*dancers\s*gay|"
    r"gay\s*couples|transgender|caitlyn\s*jenner|pizzagate)",
    re.IGNORECASE
)

# Sensitive output terms
SENSITIVE_OUTPUT_PATTERN = re.compile(
    r"(?i)\b(voyeur|rape|incest|molest|bomb|grenade|arson|meth|hack|weapon)\b",
    re.IGNORECASE
)

# Extended harmful pattern for comprehensive coverage
EXTENDED_HARMFUL_PATTERN = re.compile(
    r"(?i)(have sex|sexual|sex with|minors?|child sex|rape|molest|grooming|"
    r"incest|hook\s*up|fuck|oral sex|anal sex|penetrat(e|ion)|"
    r"watch.*nude|watch.*naked|take.*naked pictures|take.*nude pictures|"
    r"spy on|voyeur|secretly watch|peep on|"
    r"kill|stab|shoot|strangle|choke|poison|build a bomb|make a weapon|"
    r"molotov|accelerant|flammable liquids?|ignite|fire starter|"
    r"buy drugs online|illegal drugs|cocaine|methamphetamine|heroin|"
    r"drug recipe|make drugs|synthesize drugs|"
    r"dark web|onion links?|tor browser|hidden service|exploit|malware|"
    r"identity theft|scam|fraud|fake id|counterfeit|money laundering|"
    r"impersonat(e|ion)|doxx|doxxing|reveal.*address|find.*social security|"
    r"private information)",
    re.IGNORECASE
)


# =============================================================================
# Action: Check AMD Regex
# =============================================================================
@action(name="check_amd_regex")
async def check_amd_regex(
    text: str,
    check_type: str = "input"
) -> bool:
    """
    Check if text contains AMD sensitive technical information.

    Args:
        text: The text to check
        check_type: "input" or "output"

    Returns:
        True if sensitive content detected (should block), False otherwise
    """
    if not text:
        return False

    match = AMD_SENSITIVE_PATTERN.search(text)
    if match:
        log.warning(f"AMD sensitive info detected in {check_type}: {match.group()}")
        return True

    return False


# =============================================================================
# Action: Check Harmful Regex (Input)
# =============================================================================
@action(name="check_harmful_regex")
async def check_harmful_regex(
    text: str
) -> bool:
    """
    Check if input text contains harmful keywords.

    Args:
        text: The user input text to check

    Returns:
        True if harmful content detected (should block), False otherwise
    """
    if not text:
        return False

    # Check primary harmful pattern
    match = HARMFUL_INPUT_PATTERN.search(text)
    if match:
        log.warning(f"Harmful content detected in input: {match.group()}")
        return True

    # Check extended harmful pattern for comprehensive coverage
    match = EXTENDED_HARMFUL_PATTERN.search(text)
    if match:
        log.warning(f"Extended harmful content detected in input: {match.group()}")
        return True

    return False


# =============================================================================
# Action: Check Sensitive Regex (Output)
# =============================================================================
@action(name="check_sensitive_regex")
async def check_sensitive_regex(
    text: str
) -> bool:
    """
    Check if output text contains sensitive terms.

    Args:
        text: The bot output text to check

    Returns:
        True if sensitive content detected (should block), False otherwise
    """
    if not text:
        return False

    match = SENSITIVE_OUTPUT_PATTERN.search(text)
    if match:
        log.warning(f"Sensitive content detected in output: {match.group()}")
        return True

    return False


# =============================================================================
# Action: Combined Safety Check
# =============================================================================
@action(name="check_all_regex_patterns")
async def check_all_regex_patterns(
    text: str,
    check_type: str = "input"
) -> dict:
    """
    Run all regex pattern checks on the text.

    Args:
        text: The text to check
        check_type: "input" or "output"

    Returns:
        Dictionary with check results
    """
    results = {
        "amd_sensitive": False,
        "harmful": False,
        "sensitive": False,
        "blocked": False,
        "matched_patterns": []
    }

    if not text:
        return results

    # AMD check
    if AMD_SENSITIVE_PATTERN.search(text):
        results["amd_sensitive"] = True
        results["matched_patterns"].append("amd_sensitive")

    # Harmful check (primarily for input)
    if check_type == "input":
        if HARMFUL_INPUT_PATTERN.search(text) or EXTENDED_HARMFUL_PATTERN.search(text):
            results["harmful"] = True
            results["matched_patterns"].append("harmful")

    # Sensitive check (primarily for output)
    if check_type == "output":
        if SENSITIVE_OUTPUT_PATTERN.search(text):
            results["sensitive"] = True
            results["matched_patterns"].append("sensitive")

    # Set blocked flag if any check failed
    results["blocked"] = any([
        results["amd_sensitive"],
        results["harmful"],
        results["sensitive"]
    ])

    return results


# =============================================================================
# Action: Intent Classification Helper
# =============================================================================
@action(name="classify_intent_regex")
async def classify_intent_regex(
    text: str
) -> Optional[str]:
    """
    Simple regex-based intent classification for common patterns.
    This supplements the LLM-based intent classification.

    Args:
        text: The user input text

    Returns:
        Intent name if matched, None otherwise
    """
    text_lower = text.lower()

    # Banking intents
    if re.search(r"(activate|activation).*card", text_lower):
        return "activate_my_card"
    if re.search(r"(lost|stolen|missing).*card", text_lower):
        return "lost_or_stolen_card"
    if re.search(r"change.*(pin|password)", text_lower):
        return "change_pin"
    if re.search(r"(atm|cash.*machine)", text_lower):
        return "atm_support"
    if re.search(r"(order|status|tracking)", text_lower):
        return "check_order_status"

    # Small talk
    if re.search(r"^(hi|hello|hey|greetings)\b", text_lower):
        return "smalltalk_greetings_hello"
    if re.search(r"^(bye|goodbye|see you|later)\b", text_lower):
        return "smalltalk_greetings_bye"
    if re.search(r"(thank|thanks)", text_lower):
        return "smalltalk_appraisal_thank_you"
    if re.search(r"(are you|you are).*(robot|bot|ai|human)", text_lower):
        return "smalltalk_agent_chatbot"

    return None
