#  Legal Clause Extraction using CUAD

This project builds a system to automatically extract specific legal clauses from contracts using the **CUAD (Contract Understanding Atticus Dataset)**. It supports both **zero-shot prompting with LLMs** and **fine-tuning smaller QA models** like `roberta-base`.

---

##  Dataset

- **Source**: [CUAD on HuggingFace](https://huggingface.co/datasets/cuad)
- **Content**: 510 real-world contracts with labeled spans for 41 legal clause types.
- **Examples of clause types**: Confidentiality, Termination, Indemnification, Governing Law, etc.

---

##  Objective

Given a legal contract and a clause type (e.g., "Termination"), the model should extract the **exact span of text** that matches the clause.

---

##  Project Workflow

### 1. Data Exploration
- Load CUAD and explore contract lengths, label distributions, clause frequencies.

### 2. Scope Definition
- Start with a small set of clause types.
- Choose method: Zero-shot prompting or fine-tuned QA model.

### 3. Preprocessing
- Chunk long contract texts (~512-1024 tokens).
- Maintain character offset mapping for evaluation.

### 4. Baseline Inference
- Use chunk-wise prompting to extract clause text.
- Merge results across chunks.

### 5. Evaluation
- Compare predicted spans to ground truth:
  - Exact match accuracy
  - Overlap metrics: token-level F1, IoU

### 6. Model Training (Optional)
- Fine-tune `roberta-base-squad2` on CUAD-converted QA format.

### 7. Output & Deployment
- Notebook-based inference with selected clause type.
- Example predictions and evaluation logs included.

---

##  Notebooks

- `CUAD_Exploration.ipynb`: Initial data loading, cleaning, and chunking.
- `CUAD_Modeling.ipynb`: Baseline inference, model fine-tuning, evaluation.

---

##  Dependencies

- Python 3.10+
- `transformers`
- `datasets`
- `scikit-learn`
- `torch`
- `pandas`
- `matplotlib` (optional for EDA)

Install with:

```bash
pip install -r requirements.txt
"# Legal-Clause-Extraction" 
