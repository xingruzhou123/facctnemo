# NeMo Guardrails - SafeGuard Mimic Configuration

This configuration mimics the flow from the reference `safeguard-rails.yml` file, adapted for NeMo Guardrails.

## Configuration Files

| File | Purpose |
|------|---------|
| `config.yml` | Main configuration (models, rails, intents) |
| `prompts.yml` | LLM prompts for safety checks |
| `rails.co` | Colang 2.x flow definitions |
| `actions.py` | Custom regex-based actions |
| `test_config.py` | Test script to verify configuration |

## Rail Flow Comparison

### Input Rails

| SafeGuard | NeMo Guardrails Equivalent |
|-----------|---------------------------|
| `block_sensitive_amd_info` (regex) | `check amd sensitive info input` (custom action) |
| `input_harmful_regex` (regex) | `check harmful regex input` (custom action) |
| `hate_speech_detector` (llm_check) | `content safety check input $model=llama_guard` |
| `llama_guard_fast` (llm_check) | Included in content safety check |
| `llama_guard_reasoned` (llm_check) | `self check input` with custom prompt |
| `llm_intent_classifier` | `user_messages` in config.yml |

### Output Rails

| SafeGuard | NeMo Guardrails Equivalent |
|-----------|---------------------------|
| `block_sensitive_amd_info_output` (regex) | `check amd sensitive info output` (custom action) |
| `output_llama_guard_fast` (llm_check) | `content safety check output $model=llama_guard` |
| `output_llama_social_harm` (llm_check) | Included in content safety prompts |
| `output_sensitive_regex` (regex) | `check sensitive regex output` (custom action) |
| `self_consistency_check` | Not directly available (use custom action) |
| `rag_faithfulness_check` | `self check facts` |

## Usage

### 1. Install NeMo Guardrails

```bash
pip install nemoguardrails
```

### 2. Set API Keys

```bash
# For NVIDIA NIM endpoints
export NVIDIA_API_KEY="your-api-key"

# Or for OpenAI
export OPENAI_API_KEY="your-api-key"
```

### 3. Test Configuration

```bash
cd /path/to/safeguard_mimic
python test_config.py
```

### 4. Use in Python

```python
from nemoguardrails import RailsConfig, LLMRails

# Load configuration
config = RailsConfig.from_path("./safeguard_mimic")

# Create rails instance
rails = LLMRails(config)

# Generate response with guardrails
response = await rails.generate_async(
    messages=[{"role": "user", "content": "Hello, how are you?"}]
)
print(response["content"])
```

### 5. Run as Server

```bash
nemoguardrails server --config ./safeguard_mimic
```

## Customization

### Adding New Regex Patterns

Edit `actions.py` to add new patterns:

```python
NEW_PATTERN = re.compile(r"your-pattern-here", re.IGNORECASE)

@action(name="check_new_pattern")
async def check_new_pattern(text: str) -> bool:
    return bool(NEW_PATTERN.search(text))
```

Then add to `rails.co`:

```colang
flow check new pattern input
  $result = execute check_new_pattern(text=$user_message)
  if $result == True
    bot say "Blocked message"
    abort
```

### Adding New Intents

Edit `config.yml` under `user_messages`:

```yaml
user_messages:
  new_intent:
    - "example message 1"
    - "example message 2"
```

### Changing Safety Models

Edit `config.yml` to use different models:

```yaml
models:
  - type: llama_guard
    engine: nim
    model: meta/llama-guard-3-8b  # Larger model

  # Or use NVIDIA's content safety model
  - type: content_safety
    engine: nim
    model: nvidia/llama-3.1-nemoguard-8b-content-safety
```

## Features Not Directly Supported

Some SafeGuard features require custom implementation in NeMo Guardrails:

1. **Self-Consistency Check**: Requires custom action to generate multiple responses and compare
2. **RAG Confidence Threshold**: Use `search_threshold` in embedding config
3. **Intent Entities Extraction**: Requires custom LLM prompting or NER

## Differences from SafeGuard

1. **Flow-based Architecture**: NeMo uses Colang flows instead of declarative rail lists
2. **Prompt Templates**: NeMo uses Jinja2 templates in `prompts.yml`
3. **Action System**: Custom logic implemented as Python actions
4. **Model Configuration**: Models specified separately with `type` identifiers
