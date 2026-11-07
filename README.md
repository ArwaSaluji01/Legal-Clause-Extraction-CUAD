# Legal Clause Extraction using CUAD

An extractive question-answering system for identifying and extracting legal clauses from commercial contracts using **RoBERTa**, **PyTorch**, and the **CUAD dataset**.

The project fine-tunes a pretrained question-answering model to locate relevant contract passages across 41 legal clause categories, including liability, termination, licensing, confidentiality-related restrictions, and intellectual property.

## Project Overview

Commercial contracts contain important clauses spread across lengthy documents. Manually locating these clauses can be time-consuming.

This project approaches the problem as **extractive question answering**. Given a contract and a clause-specific question, the model predicts the relevant text span or returns no answer when the clause is not identified.

Rather than training a separate classifier for each category, the system uses a shared question-answering model across all 41 CUAD clause categories.

## Key Features

- Extractive legal clause identification across 41 categories.
- RoBERTa-based question-answering architecture.
- Sliding-window tokenization for long contracts.
- Context-aware answer span prediction.
- No-answer handling for missing clauses.
- Contract-level train, validation, and test separation.
- Exact Match and token-level F1 evaluation.
- GPU training with mixed precision.
- Periodic checkpointing and training recovery support.
- Standalone inference and evaluation scripts.

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Deep Learning | PyTorch |
| NLP Model | RoBERTa |
| Pretrained Checkpoint | `deepset/roberta-base-squad2` |
| NLP Framework | Hugging Face Transformers |
| Tokenization | Fast RoBERTa tokenizer |
| Dataset Processing | JSON, JSONL |
| Evaluation | Exact Match, Token-level F1 |
| Training Hardware | Google Colab NVIDIA Tesla T4 |
| Development | VS Code |

## Dataset

The project uses the **Contract Understanding Atticus Dataset (CUAD)**, introduced for legal contract review and clause identification.

Dataset: [The Atticus Project — CUAD](https://github.com/TheAtticusProject/cuad)

The dataset contains 510 commercial contracts and annotations covering 41 clause categories.

### Dataset split

| Partition | Contracts | Processed examples |
|---|---:|---:|
| Training | 367 | 20,167 |
| Validation | 41 | 2,283 |
| Official test | 102 | 4,182 |
| **Total** | **510** | **26,632** |

The original 408-contract training partition was divided into training and validation subsets. The official 102-contract test partition was kept separate.

Splitting was performed at the contract level to prevent documents from appearing across multiple partitions.

### The 41 clause categories

1. Affiliate License-Licensee
2. Affiliate License-Licensor
3. Agreement Date
4. Anti-Assignment
5. Audit Rights
6. Cap on Liability
7. Change of Control
8. Competitive Restriction Exception
9. Covenant Not to Sue
10. Document Name
11. Effective Date
12. Exclusivity
13. Expiration Date
14. Governing Law
15. IP Ownership Assignment
16. Insurance
17. Irrevocable or Perpetual License
18. Joint IP Ownership
19. License Grant
20. Liquidated Damages
21. Minimum Commitment
22. Most Favored Nation
23. No-Solicit of Customers
24. No-Solicit of Employees
25. Non-Compete
26. Non-Disparagement
27. Non-Transferable License
28. Notice Period to Terminate Renewal
29. Parties
30. Post-Termination Services
31. Price Restrictions
32. Renewal Term
33. Revenue/Profit Sharing
34. Rofr/Rofo/Rofn
35. Source Code Escrow
36. Termination for Convenience
37. Third Party Beneficiary
38. Uncapped Liability
39. Unlimited/All-You-Can-Eat-License
40. Volume Restriction
41. Warranty Duration

## Model Architecture

**Base model:** `deepset/roberta-base-squad2`

The pretrained RoBERTa question-answering model is fine-tuned on CUAD-derived question-answer examples.

For each clause category, a corresponding question is paired with the contract text. The model predicts start and end token positions representing the extracted answer.

When no relevant answer is available, the model can predict the no-answer position.

### Long-document handling

Commercial contracts can exceed the model's maximum input length. To handle this, the project uses sliding-window tokenization.

| Parameter | Value |
|---|---:|
| Maximum sequence length | 384 |
| Document stride | 128 |
| Tokenizer | Fast tokenizer |
| Truncation strategy | `only_second` |
| Answer representation | Start and end token positions |

The tokenizer creates overlapping document windows and aligns annotated answer spans with token positions.

This allows the model to process long contracts without treating an entire contract as a single sequence.

## Training

Training was performed on a Google Colab NVIDIA Tesla T4 GPU.

The training pipeline includes:

- Bounded-buffer shuffling of training examples.
- Lazy tokenization and feature selection.
- Positive-window retention for answerable examples.
- Limited negative-window sampling.
- Sampling of no-answer examples.
- FP16 mixed-precision training.
- Gradient accumulation and gradient clipping.
- Periodic checkpoint saving.
- Validation after each epoch.
- Best-checkpoint selection based on validation loss.

### Training configuration

| Parameter | Value |
|---|---:|
| Base model | `deepset/roberta-base-squad2` |
| Epochs | 2 |
| Batch size | 4 |
| Gradient accumulation | 2 |
| Effective batch size | 8 |
| Mixed precision | FP16 |
| Checkpoint interval | 250 optimizer steps |
| Training features per epoch | 5,749 |
| Validation features | 564 |

The original processed dataset contains 20,167 training examples and 2,283 validation examples. The optimized training pipeline uses a bounded selection of examples and features rather than expanding every example into every possible document window.

### Training history

| Epoch | Training loss | Validation loss | Global step |
|---|---:|---:|---:|
| 1 | 0.9867 | 0.9856 | 719 |
| 2 | 0.6970 | 1.0386 | 1,438 |

The best checkpoint was selected from **epoch 1**, with a validation loss of **0.9856**.

The decrease in training loss alongside an increase in validation loss during epoch 2 suggests that further training would require careful monitoring for overfitting.

## Evaluation

The selected best checkpoint was evaluated on all 4,182 examples in the official CUAD test partition.

### Overall results

| Metric | Result |
|---|---:|
| Evaluated examples | 4,182 |
| Exact Match | **58.82%** |
| Token-level F1 | **63.66%** |

These are results obtained using this project's evaluation pipeline and configuration. They are not claimed as official CUAD benchmark or leaderboard results.

### Evaluation methodology

- Evaluation partition: official test.
- Checkpoint: best validation-loss checkpoint.
- No-answer threshold: 0.0.
- Exact Match: measures whether the normalized prediction matches a reference answer.
- F1: measures token-level overlap between predicted and reference answers.
- Multiple gold answers and no-answer cases are handled by the evaluation utilities.

### Per-category results

| Clause category | Exact Match | F1 |
|---|---:|---:|
| Affiliate License-Licensee | 54.90% | 56.68% |
| Affiliate License-Licensor | 56.86% | 58.84% |
| Agreement Date | 78.43% | 80.12% |
| Anti-Assignment | 54.90% | 67.07% |
| Audit Rights | 59.80% | 67.73% |
| Cap on Liability | 28.43% | 39.80% |
| Change of Control | 23.53% | 29.79% |
| Competitive Restriction Exception | 47.06% | 49.58% |
| Covenant Not to Sue | 70.59% | 73.76% |
| Document Name | 66.67% | 84.17% |
| Effective Date | 44.12% | 48.19% |
| Exclusivity | 39.22% | 46.18% |
| Expiration Date | 22.55% | 32.45% |
| Governing Law | 69.61% | 86.09% |
| IP Ownership Assignment | 21.57% | 26.85% |
| Insurance | 82.35% | 87.27% |
| Irrevocable or Perpetual License | 51.96% | 54.77% |
| Joint IP Ownership | 73.53% | 76.04% |
| License Grant | 58.82% | 67.57% |
| Liquidated Damages | 81.37% | 83.44% |
| Minimum Commitment | 53.92% | 61.35% |
| Most Favored Nation | 81.37% | 81.40% |
| No-Solicit of Customers | 88.24% | 90.40% |
| No-Solicit of Employees | 79.41% | 82.54% |
| Non-Compete | 63.73% | 69.47% |
| Non-Disparagement | 94.12% | 94.70% |
| Non-Transferable License | 40.20% | 44.41% |
| Notice Period to Terminate Renewal | 75.49% | 78.69% |
| Parties | 65.69% | 73.76% |
| Post-Termination Services | 10.78% | 17.30% |
| Price Restrictions | 91.18% | 91.18% |
| Renewal Term | 62.75% | 68.01% |
| Revenue/Profit Sharing | 53.92% | 61.81% |
| Rofr/Rofo/Rofn | 41.18% | 43.13% |
| Source Code Escrow | 90.20% | 90.20% |
| Termination for Convenience | 28.43% | 32.85% |
| Third Party Beneficiary | 72.55% | 73.27% |
| Uncapped Liability | 30.39% | 32.99% |
| Unlimited/All-You-Can-Eat-License | 93.14% | 93.17% |
| Volume Restriction | 45.10% | 47.66% |
| Warranty Duration | 63.73% | 65.31% |

## Repository Structure

```text
Legal-Clause-Extraction-CUAD/
│
├── config/
│   └── config.py
│
├── src/
│   ├── dataset/
│   │   ├── download_and_inspect.py
│   │   ├── prepare_dataset.py
│   │   └── validate_processed.py
│   │
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   └── qa_utils.py
│   │
│   ├── inference/
│   │   └── predict.py
│   │
│   ├── model/
│   │   └── qa_model.py
│   │
│   ├── preprocessing/
│   │   ├── tokenizer.py
│   │   └── validate_tokenization.py
│   │
│   └── training/
│       ├── profile_pipeline.py
│       ├── train.py
│       └── validate_checkpoint.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

Generated datasets, model checkpoints, logs, and evaluation outputs are intentionally excluded from the Git repository.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/ArwaSaluji01/Legal-Clause-Extraction-CUAD.git
cd Legal-Clause-Extraction-CUAD
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows PowerShell**
```powershell
.venv\Scripts\Activate.ps1
```

**Linux/macOS**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Prepare the dataset

Run the dataset download and inspection script:

```bash
python -m src.dataset.download_and_inspect
```

Prepare the contract-level training, validation, and test partitions:

```bash
python -m src.dataset.prepare_dataset
```

Validate the processed data:

```bash
python -m src.dataset.validate_processed
```

### 5. Train the model

The full training pipeline is designed for GPU execution. Google Colab with a compatible CUDA-enabled GPU is recommended.

Example training command:

```bash
python -m src.training.train \
    --epochs 2 \
    --train-examples 4000 \
    --validation-examples 400 \
    --max-train-features 8000 \
    --max-validation-features 800 \
    --batch-size 4 \
    --gradient-accumulation 2 \
    --checkpoint-every 250
```

The training script supports checkpoint saving and resumption. Run the following to inspect available arguments:

```bash
python -m src.training.train --help
```

### 6. Evaluate

After placing a trained checkpoint in the expected checkpoint directory, run:

```bash
python -m src.evaluation.evaluate --help
```

Example evaluation command:

```bash
python -m src.evaluation.evaluate \
    --limit 0 \
    --threshold 0.0 \
    --checkpoint models/checkpoints/best_model
```

The evaluation script generates aggregate and per-category metrics along with prediction outputs.

### 7. Run inference

The standalone inference script can be used with a trained checkpoint:

```bash
python -m src.inference.predict --help
```

Refer to the available arguments for supplying a contract, selecting a clause category, and specifying the checkpoint.

**Note:** Trained model weights are not included in this repository. You must train the model or provide a compatible saved checkpoint before running evaluation or inference.

## Limitations

- The model is trained on the CUAD dataset and may not generalize equally well to every contract type or legal domain.
- Performance varies substantially across clause categories.
- Long, complex clauses and clauses with ambiguous boundaries can be difficult to extract.
- The current training pipeline uses selected features to keep GPU training computationally manageable.
- The model produces text spans, not legal interpretations or legal advice.
- Predictions should be reviewed by a qualified legal professional before being used in any real contractual decision.

## Future Improvements

- Improve performance on weaker clause categories.
- Experiment with longer context windows and alternative document chunking strategies.
- Calibrate no-answer thresholds using validation data.
- Investigate category-specific question formulation.
- Compare alternative pretrained extractive QA models.
- Improve error analysis for boundary detection and long clause extraction.

## Acknowledgements

- [The Atticus Project — CUAD](https://github.com/TheAtticusProject/cuad) for the commercial contract dataset.
- [Hugging Face](https://huggingface.co/) for the Transformers ecosystem.
- [deepset](https://huggingface.co/deepset/roberta-base-squad2) for the pretrained RoBERTa SQuAD 2.0 checkpoint.

---

**Project:** Legal Clause Extraction using CUAD  
**Task:** Extractive Question Answering  
**Model:** RoBERTa-base fine-tuned for SQuAD 2.0  
**Framework:** PyTorch and Hugging Face Transformers
