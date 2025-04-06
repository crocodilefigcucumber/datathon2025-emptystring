# Discrepancy Detection in Client Profiles
Team Name:

Team Members: Noah Stäuble, Mikael Makonnen, Michał Mikuta, Elias Mbarek

### Introduction

Client onboarding often involves collecting the same information across multiple forms—ID documents, profiles, application forms, and descriptions. Inconsistent entries across these documents (e.g. mismatched phone numbers, conflicting text descriptions) are common and hard to detect manually.

Our solution automates discrepancy detection across structured and free-form data using a layered ensemble approach, ensuring more reliable onboarding decisions in real-world banking pipelines.

### Explainability

A key strength of our solution is its explainability. Each rejection decision made by the ensemble can be fully traced back to its contributing components. Whether it's a rule-based mismatch, an LLM-detected inconsistency, or an ML classifier signal, the ensemble provides a clear breakdown of the evidence leading to the final decision. This transparency ensures trust and accountability in real-world applications.

---

### Approach Overview

We designed a modular pipeline that captures inconsistencies through multiple complementary layers:

- **Rule-Based Matching:** Symbolic comparison of structured fields (e.g. phone number, nationality, address) across forms  
- **LLM-Based Detection:** Uses a language model to detect semantic and textual inconsistencies in free-form client descriptions  
- **ML Classifiers:** Supervised models trained to detect subtle data patterns and learn from intermediate signals  
- **Ensemble Aggregator:** Final decision is made by combining all sources of evidence in a robust way  

This approach offers:
- High precision on hard symbolic logic  
- Robust recall via ML  
- Generalization to novel inconsistencies via LLMs  

---

### Setup Instructions

Set up environment using Conda:

```bash
conda create -n datathon-env python=3.10
conda activate datathon-env
pip install -r requirements.txt
```