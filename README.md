# Foundations of Generative AI (ICT4403) — Assignment 2
## Autoregressive Word-Level Text Generation with Recurrent Neural Networks

This repository contains a complete, end-to-end implementation and empirical comparative study of four recurrent neural network architectures for autoregressive word-level language modeling:
1. **Simple RNN** (Elman RNN with $\tanh$ non-linearity)
2. **Bidirectional RNN (BiRNN)** (Concatenated forward & backward states)
3. **Long Short-Term Memory (LSTM)** (Gated memory cells with cell state $c_t$)
4. **Gated Recurrent Unit (GRU)** (Coupled reset and update gates)

---

## Repository Structure

```text
fgai-autoregressive-textgen/
├── .gitignore
├── requirements.txt
├── notebook.ipynb                  # Single end-to-end Jupyter Notebook with all outputs saved
├── README.md                       # Comprehensive project documentation and empirical findings
├── data/
│   └── sherlock_holmes.txt         # Cleaned corpus text (Gutenberg #1661 stripped of licenses)
├── src/
│   ├── __init__.py
│   ├── data_utils.py               # Tokenization, vocabulary, sliding windows, DataLoaders
│   ├── models.py                   # PyTorch architectures (SimpleRNN, BiRNN, LSTM, GRU)
│   ├── train_utils.py              # Training loop, evaluation, GPU synchronization, early stopping
│   ├── generate.py                 # Autoregressive generation, temperature sampling, text metrics
│   ├── plot_utils.py               # Loss curves, bar charts, temperature quality plots
│   ├── run_experiments.py          # CLI runner for fast/full experimental workflows
│   ├── build_notebook.py           # Programmatic notebook generator with structured cells
│   └── create_submission_doc.py    # Formats university Word submission docx via python-docx
├── results/
│   ├── metrics.csv                 # Test loss, perplexity, accuracy, parameters, runtime
│   ├── text_metrics.csv            # Distinct-1, Distinct-2, Repeated Trigram, Unique Word ratios
│   ├── top20_words.png             # Distribution plot of the 20 most frequent tokens
│   ├── loss_curves.png             # Side-by-side training and validation loss curves
│   ├── perplexity_bar.png          # Test perplexity bar comparison
│   ├── accuracy_bar.png            # Top-1 vs. Top-5 test accuracy grouped bar chart
│   ├── train_time_bar.png          # GPU training duration comparison
│   ├── temperature_metrics.png     # 4-panel text quality vs. temperature plot
│   └── temperature_plots/          # Individual high-resolution temperature curves
├── generated_samples/
│   ├── samples.md                  # Markdown report of all 48 generated sample continuations
│   └── samples.csv                 # Tabular dataset of model x seed x temperature generations
└── submission/
    └── YourName_RegNum_FGAI_Assignment2.docx # University submission document
```

---

## 1. Problem Statement and Objective

Autoregressive language modeling frames text generation as a sequential next-token prediction problem over a discrete vocabulary $\mathcal{V}$:
$$P(w_1, w_2, \dots, w_T) = \prod_{t=1}^T P(w_t \mid w_1, w_2, \dots, w_{t-1})$$

In this assignment, we examine:
- How architectural inductive biases (unidirectional recurrence, bidirectional recurrence, additive cell memory in LSTMs, and gating in GRUs) affect sequential next-word prediction.
- The trade-offs between parameter efficiency, GPU training dynamics, vanishing gradients, and validation loss convergence.
- How sampling temperature ($T \in \{0.5, 1.0, 1.5\}$) governs the Pareto frontier between lexical diversity and syntactic coherence during multi-step inference.

---

## 2. Dataset Description and Source Link

- **Corpus Title:** *The Adventures of Sherlock Holmes* by Sir Arthur Conan Doyle (1892)
- **Source:** Project Gutenberg (EBook #1661)
- **Exact Download URL:** [https://www.gutenberg.org/cache/epub/1661/pg1661.txt](https://www.gutenberg.org/cache/epub/1661/pg1661.txt)

### Preprocessing & Corpus Statistics
The Project Gutenberg legal preamble and postamble were cleanly stripped using verified boundary markers (`*** START OF THE PROJECT GUTENBERG EBOOK` and `*** END OF THE PROJECT GUTENBERG EBOOK`).

| Corpus Attribute | Value |
| :--- | :--- |
| **Total Characters** | 574,123 characters |
| **Total Tokens** | 121,193 tokens |
| **Unique Tokens** | 7,891 tokens |
| **Total Sentences** | 7,291 sentences |
| **Sentence Length (Mean $\pm$ Std)** | 15.60 $\pm$ 11.45 tokens |
| **Sentence Length (Median)** | 13.0 tokens |
| **Sentence Length Range** | [1, 142] tokens |
| **Vocabulary Size** | 7,893 tokens (includes `<pad>` = 0, `<unk>` = 1) |

**Tokenization Strategy:** All text is lowercased. Words and contractions (e.g. *don't*, *sherlock's*) are preserved, and punctuation marks (`.`, `,`, `;`, `:`, `!`, `?`) are isolated as individual tokens.

![Top 20 Frequent Tokens](results/top20_words.png)

### Sliding Window Configuration & Splits
- **Context Length (`SEQ_LEN`):** 25 tokens.
- **Target:** The 26th token ($w_{t+25}$).
- **Split Ratio:** 80% Train, 10% Validation, 10% Test.
- **Partitioning Method:** Sequential chronological partitioning of the token stream to prevent window overlap leakage between splits.
  - **Train Windows:** 96,929 windows (shuffled each epoch)
  - **Validation Windows:** 12,094 windows
  - **Test Windows:** 12,095 windows

---

## 3. Models Implemented

All four models follow the standardized topology:
$$\text{Input Window } \mathbf{x}_{1:25} \xrightarrow{\text{Embedding}} \mathbf{E} \in \mathbb{R}^{B \times 25 \times 128} \xrightarrow{\text{Recurrent Layer}} \mathbf{H} \xrightarrow{\text{Dropout}(0.3)} \mathbf{h}_{25} \xrightarrow{\text{Linear Projection}} \mathbf{z} \in \mathbb{R}^{B \times 7893}$$

### Architecture Descriptions
1. **SimpleRNN (`nn.RNN`):** Classical Elman network with $\tanh$ activation. Updates hidden state via $h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh})$. Lacks gating, making it vulnerable to vanishing gradients over sequence length 25.
2. **Bidirectional RNN (`BiRNN`):** Employs two independent recurrent passes: forward ($\overrightarrow{h}_t$) and backward ($\overleftarrow{h}_t$). The concatenated final representation $[\overrightarrow{h}_{25}; \overleftarrow{h}_{25}] \in \mathbb{R}^{512}$ is projected to the vocabulary.
3. **LSTM (`nn.LSTM`):** Uses input, forget, and output gates with an additive cell state ($c_t$) to provide uninterrupted gradient highways across long temporal lags.
4. **GRU (`nn.GRU`):** Simplifies the gating mechanism into reset ($r_t$) and update ($z_t$) gates, eliminating the separate cell state while maintaining robust gradient propagation.

### Standardized Hyperparameters & Parameter Breakdown

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Embedding Dimension** | 128 | Shared lookup table size |
| **Hidden Units** | 256 | Hidden layer dimension ($2 \times 256 = 512$ for BiRNN) |
| **Recurrent Layers** | 1 | Single-layer recurrent depth |
| **Dropout Rate** | 0.3 | Applied to final hidden state prior to linear projection |
| **Optimizer** | Adam | Learning rate $\eta = 1 \times 10^{-3}$ |
| **Gradient Clipping** | 1.0 | Maximum gradient norm threshold |
| **Batch Size** | 128 | Chosen for modest GPU VRAM footprint |
| **Max Epochs** | 15 | With early stopping (patience = 3 on validation loss) |
| **Random Seed** | 42 | Fixed across Python, NumPy, and PyTorch |

### Parameter Accounting

| Component | SimpleRNN | BiRNN | LSTM | GRU |
| :--- | :---: | :---: | :---: | :---: |
| **Embedding** ($7893 \times 128$) | 1,010,304 | 1,010,304 | 1,010,304 | 1,010,304 |
| **Recurrent Layer** | 98,816 | 197,632 | 395,264 | 296,448 |
| **Linear Classifier** | 2,028,501 | 4,049,109 | 2,028,501 | 2,028,501 |
| **Total Trainable Parameters** | **3,137,621** | **5,257,045** | **3,434,069** | **3,335,253** |

> **Key Architectural Observation:** BiRNN has **5,257,045** parameters—67% more than SimpleRNN (3.14M) and 53% more than LSTM (3.43M). This occurs because bidirectional recurrence doubles the recurrent parameters (197K vs 98K) and doubles the classifier projection input dimension ($512 \to 7,893$), requiring over 4.04 million parameters in the output linear layer alone.

---

## 4. Execution Steps and Dependencies

### Windows 11 Local Execution (Recommended)
Prerequisites: Python 3.12, NVIDIA GPU with CUDA 12.6 support.

```powershell
# 1. Clone repository and navigate to folder
git clone <YOUR_REPO_URL>
cd fgai-autoregressive-textgen

# 2. Activate virtual environment (if creating fresh)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install PyTorch with CUDA 12.6
.\.venv\Scripts\pip.exe install torch==2.14.1+cu126 --index-url https://download.pytorch.org/whl/cu126

# 4. Install remaining pinned dependencies
.\.venv\Scripts\pip.exe install -r requirements.txt

# 5. Verify CUDA availability
.\.venv\Scripts\python.exe -c "import torch; print('CUDA available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))"

# 6. Execute FAST_MODE smoke test (runs in ~5s)
.\.venv\Scripts\python.exe src\run_experiments.py --fast

# 7. Execute full training and generation pipeline
.\.venv\Scripts\python.exe src\run_experiments.py --epochs 15 --batch_size 128

# 8. Execute the complete Jupyter Notebook top to bottom
.\.venv\Scripts\python.exe -m ipykernel install --user --name fgai-venv --display-name "Python (.venv)"
.\.venv\Scripts\python.exe -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=fgai-venv notebook.ipynb
```

### Optional Google Colab Instructions
If running on Colab instead of a local GPU:
```python
# In a Colab cell:
!git clone <YOUR_REPO_URL>
%cd fgai-autoregressive-textgen
!pip install -r requirements.txt
!python src/run_experiments.py --epochs 15 --batch_size 128
```

---

## 5. Results Obtained

All metrics reported below were obtained on an NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM) with precise `torch.cuda.synchronize()` runtime measurement.

### Comprehensive Model Performance Table

| Model | Parameters | Best Val Loss | Test Loss | Test Perplexity | Top-1 Acc (%) | Top-5 Acc (%) | Epochs Trained | Avg Epoch Time | Total Train Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SimpleRNN** | 3,137,621 | 5.1839 | 5.2490 | 190.38 | 17.19% | 37.97% | 6 (Best: 3) | 5.86s | 35.19s |
| **BiRNN** | 5,257,045 | 5.2446 | 5.3136 | 203.09 | 16.14% | 37.32% | 5 (Best: 2) | 6.01s | 30.06s |
| **LSTM** | 3,434,069 | 5.0571 | 5.1283 | 168.74 | 18.41% | **39.85%** | 7 (Best: 4) | 4.58s | 32.03s |
| **GRU** | 3,335,253 | **5.0407** | **5.0919** | **162.70** | **18.73%** | 39.69% | 6 (Best: 3) | 4.85s | **29.08s** |

### Visualizations

#### Loss Trajectories
![Loss Curves](results/loss_curves.png)

#### Test Perplexity Comparison (Lower is Better)
![Perplexity Bar Chart](results/perplexity_bar.png)

#### Top-1 and Top-5 Accuracy
![Accuracy Bar Chart](results/accuracy_bar.png)

#### GPU Training Runtime
![Training Time Bar Chart](results/train_time_bar.png)

---

## 6. Model Comparison and Evaluation

### Quantitative Analysis
1. **GRU Superiority:** GRU achieved the lowest test loss (**5.0919**), lowest perplexity (**162.70**), highest top-1 accuracy (**18.73%**), and fastest total training runtime (**29.08s**). Its combined reset-update gating mechanism provides an optimal parameter-to-capacity ratio on this ~121k-token dataset.
2. **LSTM Performance:** LSTM achieved the second-best perplexity (**168.74**) and the highest top-5 accuracy (**39.85%**). Its cell state memory enables robust sequence learning, closely mirroring GRU performance.
3. **SimpleRNN Bottleneck:** SimpleRNN achieved a test perplexity of **190.38** (+17.0% worse than GRU). Without gating, vanishing gradients over the 25-step history limit effective memory retention.
4. **BiRNN Causality Mismatch:** BiRNN performed worst overall (Perplexity: **203.09**, Top-1 Acc: **16.14%**) despite carrying 5.26M parameters. In autoregressive language modeling, token generation is strictly causal. During inference on prompt prefixes, the backward recurrent pass cannot observe the true future continuation, rendering the backward weights ineffective and inducing rapid overfitting on the training distribution.

### Qualitative Analysis & Coherence Scoring (1–5 Scale)

| Model | Coherence Score | Justification Based on Generated Continuations |
| :--- | :---: | :--- |
| **SimpleRNN** | **2.5 / 5.0** | Produces syntactically valid short clauses (*"I have seen my friend and now? Asked Holmes."*), but loses semantic context within 4–5 tokens. Prone to local repetition loops at $T=0.5$ (*"I have been able to... I have been caused enough..."*). |
| **BiRNN** | **2.0 / 5.0** | Displays the highest repetition traps at $T=0.5$ (repeated trigram ratio: 0.086). Generates cyclical phrasing (*"at the house, and the commissionaire in the house... a very bad, and I have seen everything of a very bad"*). Lacks causal directionality. |
| **LSTM** | **3.8 / 5.0** | Demonstrates consistent Victorian dialogue syntax and maintains subject-verb agreement across multi-clause sentences (*"You will find it on the matter? He asked. It is a small thing, and we shall see nothing of the matter. I had no doubt that I should be able to do so."*). |
| **GRU** | **4.0 / 5.0** | Produces the most natural conversational pacing, character-appropriate lexical choices, and thematic coherence at $T=1.0$ (*"You hear your best? Yes. Then you wish it? Oh? He didn't being for his chair to fear else: his account it was also instructive? Asked Holmes."*). |

---

## 7. Sampling-Temperature Analysis

Autoregressive text continuations (60 tokens each) were generated for all 4 models across 4 fixed seed prompts at temperatures $T \in \{0.5, 1.0, 1.5\}$.

### Aggregate Lexical Diversity & Repetition Metrics

| Model | Temp ($T$) | Distinct-1 Ratio | Distinct-2 Ratio | Repeated-Trigram Ratio | Unique-Word Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SimpleRNN** | 0.5 | 0.613 | 0.877 | 0.034 | 0.661 |
| **SimpleRNN** | 1.0 | 0.758 | 0.979 | 0.000 | 0.821 |
| **SimpleRNN** | 1.5 | 0.904 | 1.000 | 0.000 | 0.937 |
| **BiRNN** | 0.5 | 0.575 | 0.818 | **0.086** | 0.614 |
| **BiRNN** | 1.0 | 0.721 | 0.996 | 0.000 | 0.804 |
| **BiRNN** | 1.5 | 0.925 | 1.000 | 0.000 | 0.955 |
| **LSTM** | 0.5 | 0.638 | 0.869 | 0.039 | 0.673 |
| **LSTM** | 1.0 | 0.758 | 0.983 | 0.000 | 0.820 |
| **LSTM** | 1.5 | 0.892 | 1.000 | 0.000 | 0.928 |
| **GRU** | 0.5 | 0.588 | 0.877 | 0.039 | 0.606 |
| **GRU** | 1.0 | 0.746 | 0.979 | 0.000 | 0.806 |
| **GRU** | 1.5 | 0.913 | 1.000 | 0.000 | 0.954 |

![Temperature Metrics](results/temperature_metrics.png)

### Temperature Behavior Breakdown
1. **Low Temperature ($T = 0.5$):**
   - Softmax logits are sharply peaked, forcing the model toward greedy selection of high-frequency words (*"the"*, *"have"*, *"been"*, *"matter"*).
   - Generates grammatical syntax but suffers from loop traps (repeated trigram ratio up to 0.086).
2. **Moderate Temperature ($T = 1.0$):**
   - Optimal trade-off between fluency and diversity. Repeated-trigram ratio drops to **0.000** across all models.
   - Distinct-1 reaches 0.72–0.76 while preserving authentic Sherlockian vocabulary (*"Baker Street"*, *"Tottenham Court Road"*, *"the Speckled Band"*).
3. **High Temperature ($T = 1.5$):**
   - Flattens probability distributions, sampling uniformly from the 7,893-word vocabulary.
   - Distinct-1 exceeds 0.90 and Distinct-2 reaches 1.00, but outputs degenerate into disjointed lexical associations (*"invariably murdering entirely boscombe corridors handle houses talk"*).

### Example Generated Samples (Seed 1: *"to sherlock holmes she is always the woman . i have seldom heard him mention her under any other name ."*)

- **GRU ($T=0.5$):**
  > *"You are very much in the matter? I have no doubt that i have only no doubt that you may go back to go. I am sure that i have no doubt that i am not very much to say that it is too. But i confess that i had gone to my acquaintance, and i"*
- **GRU ($T=1.0$):**
  > *"You hear your best? Yes. Then you wish it? Oh? He didn't being for his chair to fear else: his account it was also instructive? Asked holmes. Horner were in his finger? About his handle, too. He spoke he in his thinking and heavily back down the probable of"*
- **GRU ($T=1.5$):**
  > *"You mature emerge as fair experienced december official you mother confectioner hinted first? He didn! Being we animal of communication enough else: his account act than also shown? Before us evidently fenchurch invariably murdering? Entirely? About crate handle houses talk. Lock from confess in bowed different day to women acquaintance revealing probable narratives"*

---

## 8. Conclusion Identifying the Best-Performing Model

### Overall Winner: Gated Recurrent Unit (GRU)

**Justification Summary:**
1. **Best Generalization:** Lowest test perplexity (**162.70**) and lowest test cross-entropy loss (**5.0919**).
2. **Highest Accuracy:** Top-1 next-word accuracy of **18.73%** and competitive Top-5 accuracy of **39.69%**.
3. **Fastest Execution:** Completed 6 epochs in **29.08s**, outperforming LSTM (32.03s), SimpleRNN (35.19s), and BiRNN (30.06s).
4. **Qualitative Superiority:** Scored highest in subjective coherence (**4.0 / 5.0**), producing the most stylistically faithful continuations at $T=1.0$.

### Architectural Insights
- **Gating is Essential:** Gated architectures (GRU and LSTM) markedly outperformed SimpleRNN and BiRNN in both perplexity and fluency.
- **BiRNN is Unsuitable for Autoregressive Generation:** Bidirectionality violates causality in autoregressive inference, adding parameter bloat without predictive benefit.

### Limitations of This System
1. **Corpus Size:** With 121k tokens, word-level recurrent networks face high sample sparsity for rare vocabulary items.
2. **Word-Level Vocabulary:** Fixed 7,893-token vocabulary cannot synthesize morphological derivatives or handle out-of-vocabulary entities.
3. **Recurrent Bottleneck:** Compressing multi-sentence context into a fixed 256-dimensional vector degrades long-distance dependency tracking compared to multi-head self-attention mechanisms in modern Transformer architectures.
