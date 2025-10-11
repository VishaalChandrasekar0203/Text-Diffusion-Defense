# Training Methodology - Reference Only

⚠️ **IMPORTANT: This folder is for REFERENCE ONLY**

---

## 📚 Purpose

This folder documents **how the model was trained**, including:
- Training parameters and hyperparameters
- Dataset used
- Optimization approach
- Results achieved

**The model is PRE-TRAINED and included in the library.**  
**You do NOT need to run these scripts.**

---

## 🔒 Why Training is Disabled

To protect the integrity of the pre-trained model:
- ✅ Training API methods have been removed
- ✅ Users cannot retrain or modify the model
- ✅ Embeddings are protected
- ✅ Consistent performance guaranteed

---

## 📖 Training Methodology Used

### Parameters
- **Epochs**: 200 (optimal)
- **Learning Rate**: 0.001
- **Optimizer**: AdamW
- **Batch Size**: 32
- **Scheduler**: CosineAnnealingWarmRestarts

### Dataset
- **Source**: Hugging Face aurora-m/adversarial-prompts
- **Synthetic Data**: 30+ adversarial-clean pairs
- **Categories**: Violence, illegal, manipulation, etc.

### Results Achieved
- **Semantic Preservation**: 69.1%
- **Safety Improvement**: 0.474
- **Processing Speed**: 192ms average
- **Success Rate**: 100%

---

## 🚀 How to Use the Library (Not Train)

```python
import text_diffusion_defense as ControlDD

# Pre-trained model loads automatically
defense = ControlDD.ControlDD()

# Use it immediately
clean_text = defense.get_clean_text_for_llm(user_prompt)
```

**No training needed!**

---

## 📊 Why Pre-Trained is Better

✅ **Instant Use**: No setup or training time  
✅ **Consistent Performance**: Proven results  
✅ **Cost Savings**: No training compute costs  
✅ **Quality Guaranteed**: Optimally trained model  
✅ **Protected**: Cannot be accidentally degraded  

---

## ℹ️ For Researchers

If you want to understand the methodology:
- Read `train.py` (do not run it)
- See `../results/training/` for training results
- Check `../results/benchmarks/` for performance data
- Review hyperparameters in the code

**To cite this work:**
```bibtex
@software{text_diffusion_defense,
  title={Text Diffusion Defense},
  author={Vishaal Chandrasekar},
  year={2024}
}
```

---

**Summary: This folder shows HOW the model was trained. The model is ready to use - no training required.**

