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
# Action: Check AMD Regex (Names must end with "Action" in Colang 2.x)
# =============================================================================
@action(name="CheckAmdRegexAction")
async def check_amd_regex_action(
    text: str,
    check_type: str = "input"
) -> bool:
    """
    Check if text contains AMD sensitive technical information.
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
@action(name="CheckHarmfulRegexAction")
async def check_harmful_regex_action(
    text: str
) -> bool:
    """
    Check if input text contains harmful keywords.
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
@action(name="CheckSensitiveRegexAction")
async def check_sensitive_regex_action(
    text: str
) -> bool:
    """
    Check if output text contains sensitive terms.
    """
    if not text:
        return False

    match = SENSITIVE_OUTPUT_PATTERN.search(text)
    if match:
        log.warning(f"Sensitive content detected in output: {match.group()}")
        return True

    return False


# =============================================================================
# Keep old names for backward compatibility with tests
# =============================================================================
@action(name="check_amd_regex")
async def check_amd_regex(text: str, check_type: str = "input") -> bool:
    return await check_amd_regex_action(text, check_type)


@action(name="check_harmful_regex")
async def check_harmful_regex(text: str) -> bool:
    return await check_harmful_regex_action(text)


@action(name="check_sensitive_regex")
async def check_sensitive_regex(text: str) -> bool:
    return await check_sensitive_regex_action(text)


# =============================================================================
# Llama Guard Check Actions (matching SafeGuard llama_guard_fast)
# =============================================================================

# LlamaGuard endpoint configuration
LLAMA_GUARD_BASE_URL = "http://localhost:8002/v1"
LLAMA_GUARD_MODEL = "meta-llama/Llama-Guard-3-1B"


@action(name="call_llama_guard_input")
async def call_llama_guard_input(text: str) -> bool:
    """
    Call LlamaGuard to check if input is unsafe.
    Returns True if content is UNSAFE and should be blocked.
    """
    import httpx

    if not text:
        return False

    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Task: Check if there is unsafe content in 'User' messages in conversations according to our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
S1: Violent Crimes.
S2: Non-Violent Crimes.
S3: Sex Crimes.
S4: Child Exploitation.
S5: Defamation.
S6: Specialized Advice.
S7: Privacy.
S8: Intellectual Property.
S9: Indiscriminate Weapons.
S10: Hate.
S11: Self-Harm.
S12: Sexual Content.
S13: Elections.
S14: Code Interpreter Abuse.
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

User: {text}

<END CONVERSATION>

Provide your safety assessment for ONLY THE LAST User message in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{LLAMA_GUARD_BASE_URL}/completions",
                json={
                    "model": LLAMA_GUARD_MODEL,
                    "prompt": prompt,
                    "max_tokens": 50,
                    "temperature": 0.0,
                },
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            result = response.json()
            generated_text = result.get("choices", [{}])[0].get("text", "").strip().lower()

            log.info(f"LlamaGuard input check: '{text[:50]}...' -> {generated_text[:50]}")

            if "unsafe" in generated_text:
                log.warning(f"LlamaGuard blocked input: {text[:100]}...")
                return True
            return False

    except Exception as e:
        log.error(f"Error calling LlamaGuard for input: {e}")
        return False


@action(name="call_llama_guard_output")
async def call_llama_guard_output(text: str) -> bool:
    """
    Call LlamaGuard to check if output is unsafe.
    Returns True if content is UNSAFE and should be blocked.
    """
    import httpx

    if not text:
        return False

    prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Task: Check if there is unsafe content in 'Agent' messages in conversations according to our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
S1: Violent Crimes.
S2: Non-Violent Crimes.
S3: Sex Crimes.
S4: Child Exploitation.
S5: Defamation.
S6: Specialized Advice.
S7: Privacy.
S8: Intellectual Property.
S9: Indiscriminate Weapons.
S10: Hate.
S11: Self-Harm.
S12: Sexual Content.
S13: Elections.
S14: Code Interpreter Abuse.
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>

Agent: {text}

<END CONVERSATION>

Provide your safety assessment for ONLY THE LAST Agent message in the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{LLAMA_GUARD_BASE_URL}/completions",
                json={
                    "model": LLAMA_GUARD_MODEL,
                    "prompt": prompt,
                    "max_tokens": 50,
                    "temperature": 0.0,
                },
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            result = response.json()
            generated_text = result.get("choices", [{}])[0].get("text", "").strip().lower()

            log.info(f"LlamaGuard output check: '{text[:50]}...' -> {generated_text[:50]}")

            if "unsafe" in generated_text:
                log.warning(f"LlamaGuard blocked output: {text[:100]}...")
                return True
            return False

    except Exception as e:
        log.error(f"Error calling LlamaGuard for output: {e}")
        return False


@action(name="llama_guard_check_input")
async def llama_guard_check_input(text: str, llm=None, config=None) -> bool:
    """
    Check input text using Llama Guard model.
    Returns True if content is UNSAFE and should be blocked.
    """
    if not text:
        return False

    try:
        # Use the llama_guard model from config
        if llm is None:
            log.warning("LLM not provided for Llama Guard check, skipping")
            return False

        # Llama Guard prompt format
        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Task: Check if there is unsafe content in 'User' messages in conversations according to our safety policy.

<BEGIN CONVERSATION>

User: {text}

<END CONVERSATION>

Provide your safety assessment for the user message.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        # Call the Llama Guard model
        response = await llm.generate_async(prompt=prompt)

        if response:
            response_lower = response.lower().strip()
            # Llama Guard returns "safe" or "unsafe"
            if "unsafe" in response_lower:
                log.warning(f"Llama Guard detected unsafe input: {text[:100]}...")
                return True

        return False

    except Exception as e:
        log.error(f"Error in Llama Guard input check: {e}")
        return False


@action(name="llama_guard_check_output")
async def llama_guard_check_output(text: str, llm=None, config=None) -> bool:
    """
    Check output text using Llama Guard model.
    Returns True if content is UNSAFE and should be blocked.
    """
    if not text:
        return False

    try:
        if llm is None:
            log.warning("LLM not provided for Llama Guard check, skipping")
            return False

        # Llama Guard prompt format for output
        prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>

Task: Check if there is unsafe content in 'Agent' messages in conversations according to our safety policy.

<BEGIN CONVERSATION>

Agent: {text}

<END CONVERSATION>

Provide your safety assessment for the agent message.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

        response = await llm.generate_async(prompt=prompt)

        if response:
            response_lower = response.lower().strip()
            if "unsafe" in response_lower:
                log.warning(f"Llama Guard detected unsafe output: {text[:100]}...")
                return True

        return False

    except Exception as e:
        log.error(f"Error in Llama Guard output check: {e}")
        return False


@action(name="classify_intent_regex")
async def classify_intent_regex(text: str) -> Optional[str]:
    """
    Simple regex-based intent classification for common patterns.
    """
    text_lower = text.lower()

    # Banking intents
    if re.search(r"(activate|activation).*card", text_lower):
        return "activate_my_card"
    if re.search(r"(lost|stolen|missing).*card|card.*(lost|stolen|missing)", text_lower):
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
