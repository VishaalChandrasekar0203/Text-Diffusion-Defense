# TextDiffuse

## 📋 Overview

TextDiff is a lightweight, pre-trained library that protects Large Language Models from adversarial text attacks using embedding-based diffusion processes. It acts as a safety middleware layer between user input and LLM processing, cleaning potentially harmful prompts while preserving semantic meaning.

Unlike commercial solutions that destroy 63-71% of user intent, TextDiff maintains **69.3% semantic preservation** while providing robust safety controls. The system runs entirely locally on CPU, ensuring complete privacy and zero API costs.

**Key Innovation**: First application of diffusion models to LLM safety, achieving superior semantic preservation through embedding-space transformations.

```python
import textdiff
defense = textdiff.ControlDD()
clean_text = defense.get_clean_text_for_llm(user_prompt)
```

---

## 📦 Installation

```bash
pip install git+https://github.com/VishaalChandrasekar0203/text-diffusion-defense.git
```

---

## 🚀 Quick Start

### Simple Usage (3 Lines)
```python
import textdiff

defense = textdiff.ControlDD()
clean_text = defense.get_clean_text_for_llm(user_prompt)
# Send to your LLM
```

### Transparent Workflow
```python
import textdiff

defense = textdiff.ControlDD()
result = defense.analyze_and_respond(user_prompt)

if result['send_to_llm']:
    response = your_llm.generate(result['llm_prompt'])
    
elif result['status'] == 'needs_clarification':
    show_message(result['message_to_user'])
    user_choice = get_user_choice()  # 'original' or 'cleaned'
    
    verification = defense.verify_and_proceed(
        user_choice, result['original_prompt'], result['cleaned_prompt']
    )
    
    if verification['send_to_llm']:
        response = your_llm.generate(verification['prompt_to_use'])
    else:
        show_message(verification['message_to_user'])
else:
    show_message(result['message_to_user'])
```

---

## 🤖 Quick Integration

### OpenAI
```python
import textdiff
import openai

defense = textdiff.ControlDD()
clean = defense.get_clean_text_for_llm(user_input)

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": clean}]
)
```

### Anthropic
```python
import textdiff
import anthropic

defense = textdiff.ControlDD()
clean = defense.get_clean_text_for_llm(user_input)

client = anthropic.Client(api_key="key")
response = client.messages.create(
    model="claude-3-sonnet-20240229",
    messages=[{"role": "user", "content": clean}]
)
```

### Any LLM
```python
defense = textdiff.ControlDD()
clean = defense.get_clean_text_for_llm(user_input)
response = your_llm.generate(clean)  # Works with any LLM!
```

**More examples**: [research_details/INTEGRATION_GUIDE.md](research_details/INTEGRATION_GUIDE.md)

---

## 📊 Performance Benchmarks

| System | Safety Improvement | Semantic Preservation | Speed |
|--------|-------------------|----------------------|-------|
| **TextDiff** | 0.453 | **0.693** 🏆 | 60ms |
| OpenAI Safety | 0.690 | 0.370 | 50ms |
| Anthropic Safety | 0.710 | 0.290 | 30ms |

**TextDiff delivers 2X better semantic preservation (69.3% vs 29-37%) while maintaining robust safety controls.**

**Training Data**: 17,715 adversarial-clean pairs  
**Model**: 500K parameters, 384-dim embeddings  
**Details**: [research_details/DATASET_AND_SCALING.md](research_details/DATASET_AND_SCALING.md)

---

## 🔒 Safety Features

- **Hybrid Detection System (97.8% Coverage)**: Pattern matching (500+ words) + fuzzy matching + embedding-based semantic analysis
- **Advanced Obfuscation Detection**: Leetspeak, spacing tricks, special characters, synonyms, emerging slang, euphemisms
- **Multi-Category Detection**: 60+ patterns across 10 categories (violence, illegal, manipulation, hate, self-harm, terrorism, obfuscation, context-bypass)
- **Transparent Analysis**: Users see what was detected with clear explanations
- **Double Verification**: Re-verifies confirmations to prevent bypass attempts
- **Adaptive Thresholds**: Context-aware safety (educational, research, safety-critical)
- **Semantic Preservation**: Maintains 69.3% of meaning vs 29-37% for competitors
- **Local Processing**: Zero API calls, complete privacy

---

## 🎓 How It Works

![TextDiff Architecture](assets/TextDiff%20Arch.jpg)

1. **Input**: User prompt → 384-dim embedding
2. **Analysis**: Pattern detection (9 categories)
3. **Diffusion**: 1000-step cleaning process
4. **Preservation**: Maintains semantic meaning
5. **Output**: Safe text for LLM

All local on CPU - no external APIs.

**Technical details**: [research_details/TECHNICAL_DETAILS.md](research_details/TECHNICAL_DETAILS.md)

---

## Documentation

**Quick References:**
- **[INTEGRATION_GUIDE.md](research_details/INTEGRATION_GUIDE.md)** - Web apps, APIs, CLIs, serverless, mobile, and more (15+ examples)
- **[DATASET_AND_SCALING.md](research_details/DATASET_AND_SCALING.md)** - Training data (17,715 pairs), scaling forecasts, performance estimates
- **[TECHNICAL_DETAILS.md](research_details/TECHNICAL_DETAILS.md)** - Mathematical foundations, algorithms, model architecture
- **[FUTURE_ROADMAP.md](research_details/FUTURE_ROADMAP.md)** - Scaling plans, compute requirements, cost estimates

---

## 📂 Project Structure

```
textdiff/
├── README.md                  # Quick start (this file)
├── demo.py                    # Usage examples
├── textdiff/                  # Core library
│   ├── control_dd.py         # ControlDD class
│   ├── model.py              # Diffusion model
│   ├── utils.py              # Utilities
│   └── __init__.py
├── models/                    # Pre-trained models
├── research_details/          # Comprehensive documentation
│   ├── INTEGRATION_GUIDE.md
│   ├── DATASET_AND_SCALING.md
│   ├── TECHNICAL_DETAILS.md
│   └── FUTURE_ROADMAP.md
├── scripts/                   # Training methodology (reference)
├── tests/                     # Test suite
└── results/                   # Benchmarks
```

---

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/VishaalChandrasekar0203/text-diffusion-defense/issues)
- **Email**: vishaalchandrasekar0203@gmail.com
- **Documentation**: [research_details/](research_details/) folder

---

## Citation

```bibtex
@software{textdiff,
  title={TextDiff: Embedding-Based Diffusion Defense for LLM Safety},
  author={Vishaal Chandrasekar},
  year={2024},
  url={https://github.com/VishaalChandrasekar0203/text-diffusion-defense}
}
```

