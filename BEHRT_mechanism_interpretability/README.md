# BEHRT Mechanism Interpretability

**Status:** 🔄 Not Started (Future Work)  
**Priority:** Low (per publication strategy)

---

## 📋 Purpose

This directory is reserved for future work on understanding BEHRT model internals through interpretability methods.

---

## 🎯 Planned Work (Deprioritized)

### Attention Analysis
- Visualize attention patterns across temporal sequences
- Identify which historical events drive predictions
- Temporal dependency mapping

### Feature Attribution
- SHAP values for diagnosis code importance
- Integrated gradients for attribution
- Per-prediction explanations

### Mechanism Discovery
- What patterns does the model actually learn?
- Comorbidity relationship extraction
- Temporal causality inference

### Clinical Decision Justification
- Generate human-readable explanations
- "Why did the model predict diabetes recurrence?"
- Compliance with clinical interpretability requirements

---

## ⚠️ Why Not Started

Per `../docs/publication_readiness_summary.md`:

> **Section 7: What is not worth doing right now**
>
> Do not spend much time on:
> - deep mechanistic interpretability of the transformer
> - trying to "explain the whole model" in biological terms
> - broad architecture experimentation without a hypothesis
>
> Those directions are **lower ROI** for a thesis with limited time.

**Current Priority:** Rigorous evaluation, honest limitations, and focused clinical validation (not deep interpretability).

---

## 📊 When This Becomes Relevant

**After thesis completion:**
- If model is deployed in clinical practice
- For understanding failure modes
- For regulatory compliance
- For building clinician trust

**For publication:**
- Only surface-level interpretability needed (attention maps, top features)
- Deep mechanism discovery is future work

---

## 🔬 Potential Methods (Reference)

### Tools to Consider
- **Captum** (PyTorch interpretability library)
- **SHAP** (model-agnostic explanations)
- **Integrated Gradients**
- **Attention Rollout** (for transformers)
- **BertViz** (transformer attention visualization)

### Clinical Interpretability Standards
- TRIPOD+AI guidelines
- FDA guidance on AI/ML medical devices
- CLAIM checklist for clinical AI

---

## 📝 Notes

This folder exists to:
1. Acknowledge interpretability as important future work
2. Separate it from current thesis priorities
3. Provide a staging area when/if interpretability becomes necessary

**Current focus remains on:**
- Completing EDA (steps 4-7)
- Disease-level performance analysis
- NEW vs RECURRING stratified evaluation
- Honest limitations documentation

---

**Last Updated:** August 29, 2026  
**Status:** Placeholder for future work  
**Decision:** Focus on evaluation quality, not model interpretability (thesis timeline constraint)
