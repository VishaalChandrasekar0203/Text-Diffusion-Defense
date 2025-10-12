# Text Diffusion Defense

Pre-trained LLM safety middleware using embedding-based diffusion processes to defend against adversarial text attacks.


---

## 🎓 How It Works

1. **Input Processing**: User prompt converted to 384-dim embedding
2. **Safety Analysis**: Pattern detection identifies risk categories
3. **Diffusion Cleaning**: 1000-step process removes adversarial content
4. **Semantic Preservation**: Advanced techniques maintain meaning
5. **Safe Output**: Clean text returned for LLM processing


---

## 🔒 Safety Features

- **Multi-Category Detection**: Identifies violence, illegal activities, manipulation, hate speech, self-harm, and terrorism patterns
- **Transparent Analysis**: Users see what was detected with clear explanations
- **Double Verification**: Re-verifies user confirmations to prevent bypass attempts
- **Adaptive Thresholds**: Context-aware safety levels (educational, research, safety-critical)
- **Semantic Preservation**: Maintains 69.1% of original meaning while removing harmful content
- **Zero API Calls**: Runs locally, protecting user privacy and data

---

## 📊 Performance Benchmarks

| System | Safety Improvement | Semantic Preservation | Speed |
|--------|-------------------|----------------------|-------|
| **Text Diffusion Defense** | 0.453 | **0.693** 🏆 | 60ms |
| OpenAI Safety | 0.690 | 0.370 | 50ms |
| Anthropic Safety | 0.710 | 0.290 | 30ms |

**Text Diffusion Defense achieves 2X better semantic preservation (69.3% vs 29-37%) while maintaining robust safety controls.**

---

## 💡 Impact

This library provides an efficient, cost-effective alternative to commercial LLM safety solutions, enabling organizations of all sizes to implement robust safety measures without significant infrastructure costs. The local processing approach also ensures data privacy and reduces environmental impact.

---

## 📚 Project Structure

```
text_diffusion_defense/
├── README.md                      # This file
├── demo.py                        # Usage examples
├── text_diffusion_defense/        # Core library
│   ├── __init__.py
│   ├── control_dd.py             # Main interface
│   ├── model.py                  # Diffusion model
│   └── utils.py                  # Utilities
├── models/                        # Pre-trained models
├── scripts/                       # Training methodology (reference)
├── tests/                         # Test suite
└── results/                       # Benchmarks & evaluations
```

---

## 📦 Installation

```bash
pip install git+https://github.com/VishaalChandrasekar0203/text-diffusion-defense.git
```

---

## 🚀 Quick Start

### Simple Usage

```python
import text_diffusion_defense as ControlDD

defense = ControlDD.ControlDD()
clean_text = defense.get_clean_text_for_llm(user_prompt)
# Send clean_text to your LLM
```

---

## Workflow

```python
import text_diffusion_defense as ControlDD

defense = ControlDD.ControlDD()

# Step 1: Analyze
result = defense.analyze_and_respond(user_prompt)

if result['send_to_llm']:
    response = your_llm.generate(result['llm_prompt'])
    
elif result['status'] == 'needs_clarification':
    show_message(result['message_to_user'])
    user_choice = get_user_choice()  # 'original' or 'cleaned'
    
    # Step 2: Verify
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

## 🤖 Integration

### With OpenAI

```python
import text_diffusion_defense as ControlDD
import openai

defense = ControlDD.ControlDD()
clean_text = defense.get_clean_text_for_llm(user_input)

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": clean_text}]
)
```

### With Anthropic Claude

```python
import text_diffusion_defense as ControlDD
import anthropic

defense = ControlDD.ControlDD()
clean_text = defense.get_clean_text_for_llm(user_input)

client = anthropic.Client(api_key="your-key")
response = client.messages.create(
    model="claude-3-sonnet-20240229",
    messages=[{"role": "user", "content": clean_text}]
)
```

### With Any LLM

```python
defense = ControlDD.ControlDD()
clean_text = defense.get_clean_text_for_llm(user_input)
response = your_llm.generate(clean_text)
```

---

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/VishaalChandrasekar0203/text-diffusion-defense/issues)
- **Email**: vishaalchandrasekar0203@gmail.com

---

## 📚 Citation

```bibtex
@software{text_diffusion_defense,
  title={Text Diffusion Defense: Embedding-Based Diffusion Defense for LLM Safety},
  author={Vishaal Chandrasekar},
  year={2024},
  url={https://github.com/VishaalChandrasekar0203/text-diffusion-defense}
}
```

---

## Acknowledgments

- Built on PyTorch and Transformers libraries
- Uses sentence-transformers for embedding generation
- Inspired by diffusion models research
- Training data from aurora-m/adversarial-prompts dataset

---

