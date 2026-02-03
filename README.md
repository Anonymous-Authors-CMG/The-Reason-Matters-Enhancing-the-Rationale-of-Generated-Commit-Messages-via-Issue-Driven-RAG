# The-Reason-Matters-Enhancing-the-Rationale-of-Generated-Commit-Messages-via-Issue-Driven-RAG


## 📋 Empirical Study

📝 [Questionnaire: Generating Commit Messages with LLMs: Your voice Matters!](https://docs.google.com/forms/d/1MTgG3pYQbOPIzw46vhYYSqZ6hCk25AQ2FgBUYoIPgxg/edit?hl=zh-cn&pli=1)

### Ethics Statement

Our research strictly adheres to [GitHub's Acceptable Use Policies](https://docs.github.com/en/site-policy/acceptable-use-policies).

This study has been reviewed and approved by our institution's ethics committee:
- [Ethics Review Approval (English)](assets/ethics_review_en.png)
- [Ethics Review Approval (Chinese)](assets/ethics_review_cn.png)


## 📁 Project Structure

### 1. `data/` - Data Directory

Contains all data files for this project.

#### 1.1 `data/get_data/` - Data Collection Scripts

Scripts for collecting new data from GitHub or reproducing the dataset.

**Usage:**
1. Configure parameters in `data/get_data/config.py`:
   - `GITHUB_TOKENS`: GitHub API tokens (supports multiple tokens for rotation to avoid rate limiting)
   - `OWNER`: Repository owner
   - `REPO`: Repository name
   - `GIT_REPO_PATH`: Local Git repository path
2. Run `python data/get_data/run_all.py` to execute the complete data collection pipeline

**Pipeline:**
| Script | Description |
|--------|-------------|
| `1_get_issue.py` | Fetch Issues data |
| `2_get_issue-commit_link.py` | Extract Issue-Commit mappings |
| `3.0_get_commit_detail.py` | Retrieve Commit details |
| `3.1_remove_bot.py` | Remove Bot commits |
| `3.2_remove_4096_token_diff.py` | Filter diffs exceeding 4096 tokens |
| `4_merge_issue_and_commit.py` | Merge Issue and Commit data |

#### 1.2 `data/eval/` - Evaluation Dataset

Sampled data based on statistical working hours:
- `dl390.json`: Deep Learning frameworks (PyTorch, TensorFlow, Keras) - 390 samples
- `mq450.json`: Message Queue frameworks (Kafka, RocketMQ, Pulsar) - 450 samples

#### 1.3 `data/issue/` - Issue Data

Contains raw data collected from `data/get_data/` and their embedding vectors.

| Subdirectory | Description |
|--------------|-------------|
| `issue_base/` | Raw Issue data |
| `issue_set/` | All Issue collections extracted from `issue_base/` |
| `issue_embedding/` | Embeddings generated using OpenAI `text-embedding-ada-2` |
| `issue_embedding_large/` | Embeddings generated using OpenAI `text-embedding-3-large` |

#### 1.4 `data/input/` - Model Input Data

Preprocessed data files ready for model input.

- `{project}_TRAIN.csv` / `{project}_DEV.csv` / `{project}_TEST.csv`: Train/Dev/Test datasets
- `tsv/`: Triplet data parsed from the Code Knowledge Graph
  - `{project}_triplets.tsv`: Triplets for each project
  - `train_data.tsv` / `test_data.tsv`: Merged train/test data

### 2. `model/` - Models Directory

Contains all models used in our research.

#### 2.1 `model/TransR.py` - Graph Embedding Model

TransR model for knowledge graph embedding. Takes the knowledge graph constructed by `utils/CodeKnowledgeGraph.py` or TSV triplet files as input.

**Training:**
```bash
python TransR.py
```

#### 2.2 `model/BertModel/` - DeepBert Model

BERT-based model for Issue-Commit linking.

**Training:**
```bash
python run_pre.py --key xxx --pro xxx
```

**Optional parameters:**
- `--num_train_epochs`: Number of training epochs
- `--train_batch_size`: Training batch size
- `--learning_rate`: Learning rate

#### 2.3 `model/encoder/` - Twin-BERT Encoder

Sentence Transformer-based encoder for encoding code diffs and issues into the same embedding space.

**Training:**
```bash
python CodeIssueEncoder.py
```

#### 2.4 `model/lstm/` - Bi-LSTM Classifier

Bi-LSTM model for quickly determining whether a commit message contains "What" and "Why" information.

For more details, refer to: [What-Makes-a-Good-Commit-Message](https://github.com/WhatMakesAGoodCM/What-Makes-a-Good-Commit-Message)

**Training:**
```bash
python Bi_LSTM.py
```

#### 2.5 `model/MLP.py` - Multi-Layer Perceptron

Final MLP classifier that fuses features from BertModel and TransR for Issue-Commit link Identification.

**Training:**
```bash
python MLP.py
```

### 3. `results/` - Experimental Results

Contains all experimental results from our study.

#### 3.1 Baselines
- Zero-shot LLMs: GPT-4o, Gemini-2.5-pro, Deepseek-v3
- ERICommiter [Xue et al. TSE-2024](https://ieeexplore.ieee.org/abstract/document/10713474/)
- OMG [Li et al. ASE-2024](https://dl.acm.org/doi/abs/10.1145/3643760)

#### 3.2 Our Method (`Ours/`)

| Subdirectory | Description |
|--------------|-------------|
| `full/` | Complete method with all components |
| `issue-only/` | Ablation: Issue retrieval only (w/o RAG) |
| `rag-only/` | Ablation: RAG only (w/o Issue linking) |

#### File Naming Convention

All result files follow the pattern `{llm}_{method}.csv`:
- `gpt4o_*.csv`: Results using GPT-4o
- `gemini_*.csv`: Results using Gemini-2.5-pro
- `dsv3_*.csv`: Results using DeepSeek-V3
