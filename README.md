# TextDiff

**Pre-trained LLM safety middleware with 2X better semantic preservation.**

```python
import textdiff
defense = textdiff.ControlDD()
clean_text = defense.get_clean_text_for_llm(user_prompt)
```

---

## 🔒 Safety Features

- **Multi-Category Detection**: 33+ patterns across 9 categories (violence, illegal, manipulation, hate, self-harm, terrorism)
- **Transparent Analysis**: Users see what was detected with clear explanations
- **Double Verification**: Re-verifies confirmations to prevent bypass attempts
- **Adaptive Thresholds**: Context-aware safety (educational, research, safety-critical)
- **Semantic Preservation**: Maintains 69.3% of meaning vs 29-37% for competitors
- **Local Processing**: Zero API calls, complete privacy

---

## 📊 Performance

| System | Safety | Semantic | Speed |
|--------|--------|----------|-------|
| **TextDiff** | 0.453 | **0.693** 🏆 | 60ms |
| OpenAI | 0.690 | 0.370 | 50ms |
| Anthropic | 0.710 | 0.290 | 30ms |

**TextDiff delivers 2X better semantic preservation (69.3% vs 29-37%)** while maintaining robust safety.

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
    # Show suggestion to user
    show_message(result['message_to_user'])
    user_choice = get_user_choice()  # 'original' or 'cleaned'
    
    # Verify choice with double-check
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

## 🤖 Quick Integration Examples

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

**For more integration examples** (Flask, FastAPI, Serverless, Mobile, etc.):  
📖 **See [research_details/INTEGRATION_GUIDE.md](research_details/INTEGRATION_GUIDE.md)**

---

## 🎓 How It Works

1. **Input**: User prompt → 384-dim embedding
2. **Analysis**: Pattern detection (9 categories)
3. **Diffusion**: 1000-step cleaning process
4. **Preservation**: Maintains semantic meaning
5. **Output**: Safe text for LLM

All local on CPU - no external APIs.

**For mathematical details and algorithms:**  
📖 **See [research_details/TECHNICAL_DETAILS.md](research_details/TECHNICAL_DETAILS.md)**

---

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **README.md** | Quick start and basic usage (you are here) |
| [INTEGRATION_GUIDE.md](research_details/INTEGRATION_GUIDE.md) | Integration examples for web apps, APIs, CLIs, serverless, mobile, and more |
| [DATASET_AND_SCALING.md](research_details/DATASET_AND_SCALING.md) | Training data details, scaling forecasts, performance estimates |
| [TECHNICAL_DETAILS.md](research_details/TECHNICAL_DETAILS.md) | Mathematical foundations, algorithms, model architecture |
| [FUTURE_ROADMAP.md](research_details/FUTURE_ROADMAP.md) | Scaling plans, compute requirements, cost estimates |

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
│   ├── INTEGRATION_GUIDE.md  # Integration examples
│   ├── DATASET_AND_SCALING.md# Data & scaling info
│   ├── TECHNICAL_DETAILS.md  # Math & algorithms
│   └── FUTURE_ROADMAP.md     # Scaling plans
├── scripts/                   # Training methodology (reference)
├── tests/                     # Test suite
└── results/                   # Benchmarks
```

---

## 💡 Impact

Efficient, cost-effective alternative to commercial LLM safety solutions. Enables organizations to implement robust safety without infrastructure costs. Local processing ensures privacy and reduces environmental impact.

---

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/VishaalChandrasekar0203/text-diffusion-defense/issues)
- **Email**: vishaalchandrasekar0203@gmail.com
- **Documentation**: See [research_details/](research_details/) folder

---

## 📚 Citation

```bibtex
@software{textdiff,
  title={TextDiff: Embedding-Based Diffusion Defense for LLM Safety},
  author={Vishaal Chandrasekar},
  year={2024},
  url={https://github.com/VishaalChandrasekar0203/text-diffusion-defense}
}
```

---

## 🙏 Acknowledgments

- Built on PyTorch and Transformers
- sentence-transformers/all-MiniLM-L6-v2 for embeddings
- Inspired by diffusion models (Ho et al., 2020)
- Training: aurora-m/adversarial-prompts (17,680 pairs) + synthetic (35 pairs)

---

*TextDiff - Simple, powerful, privacy-focused LLM safety*
