# Contextualized Code Comment AutoCorrection

An NLP project that automatically detects nad corrects the tpos in code comments using a fine-tuned T5 transformer model. Supports Python and C++.

## What it does

Paste any Python or C++ code with typos in comments - the model finds and fixes them automatically. 

### Example

#### Before

```python
def get_user(id):
    """Returns the usr from databse by ID"""
    # Query the database for the user
    return db.query(id)
```

#### After

```python
def get_user(id):
    """Returns the user from database by ID"""
    # Query the database for the user
    return db.query(id)
```

## Why it is different from a regular spell checker

A regular spell checker has no understanding of code context.
This model uses the surrounding function name, variable names, and logic to make contextually appropriate corrections.

for example:
- 'get_child_folders()' + comment 'foleder' -> corrected to 'folder' (not 'feeder')
- 'db.query(id)' + comment 'Retuns the usr from databse' -> 'Returns the user from database'

## Architecture

### Model

- **T5-small** (60M parameters) fine-tuned for seq2seq spelling correction
- Input format: `"fix: <corrupted comment>"` → Output: `"<clean comment>"`
- Trained on Kaggle T4 GPU

### Why T5 over BERT/RoBERTa

We first tried RoBERTa MLM (masked language model) approach — masking the corrupted
word and asking the model to predict the correct one. This failed because masking
removes the typo entirely, so the model predicts contextually plausible words
instead of fixing spelling.

T5 is a seq2seq model — it sees the full corrupted sentence and rewrites it.
This is the correct architecture for spell correction.

### Training Data

- Source: CodeSearchNet (412,178 Python functions from GitHub)
- Extracted: 20,000 clean docstring summary lines
- Synthetic typo injection: 4 strategies
  - **Swap**: adjacent character swap (`returns` → `reutrns`)
  - **Delete**: remove one character (`returns` → `retuns`)
  - **Insert**: add a random character (`index` → `indxex`)
  - **Substitute**: replace with keyboard neighbor (`list` → `lust`)
- Training set: 15,000 examples
- Evaluation set: 5,000 examples

### Comment Extraction

Rule-based parser handles:
- Python: `#` inline comments and `"""` / `'''` docstrings
- C++: `//` inline comments and `/* */` block comments

## Project Structure

```text
autocorrect_comments/
│
├── data/
│   ├── train_data.jsonl          # 20,000 (code, corrupted, original) pairs
│   └── tokenized_data.pkl        # RoBERTa tokenized data (Approach 1)
│
├── tokenizer/                    # RoBERTa tokenizer files (Approach 1)
│
├── t5_trained/                   # Fine-tuned T5 model weights
│   ├── model.safetensors
│   ├── config.json
│   ├── generation_config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── spiece.model
│
├── trained_model/                # RoBERTa trained weights (Approach 1, deprecated)
│
├── data_pipeline.ipynb           # Step 1-2: Load CodeSearchNet, inject typos
├── 02_training.ipynb             # Step 3-4: RoBERTa approach (documented, deprecated)
├── 03_inference.ipynb            # Step 5-7: T5 inference, evaluation, Gradio demo
└── README.md
```

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `data_pipeline.ipynb` | Load CodeSearchNet, extract docstrings, inject synthetic typos, save dataset |
| `02_training.ipynb` | First approach using RoBERTa MLM — documented as lessons learned |
| `03_inference.ipynb` | T5 inference pipeline, comment extractor, full pipeline, Gradio demo |

T5 training was done on Kaggle (free T4 GPU) — see `autocorrect-T5-training` notebook on Kaggle.

## Results

| Metric | Value |
|--------|-------|
| Training examples | 15,000 |
| Training epochs | 12 |
| Model size | 60M parameters |
| Common typo fix rate | ~70% on fresh sentences |
| Supported languages | Python, C++ |

**What works well:**
- Character deletion typos: `aray` → `array`, `sequnce` → `sequence`
- Character transposition: `databse` → `database`, `Retuns` → `Returns`
- Multi-word correction in a single pass

**What does not work well:**
- Very short words (3 letters or less) — too ambiguous
- Rare patterns not seen enough in training data
- Keyboard substitution typos on uncommon words

## How to Run

### 1. Install dependencies

```bash
pip install transformers torch gradio pyspellchecker sentencepiece datasets
```

### 2. Download the trained model

Download the `t5_trained/` folder and place it at:

```text
autocorrect_comments/t5_trained/
```

Also download `spiece.model` from:

https://huggingface.co/t5-small/resolve/main/spiece.model

Place it inside `t5_trained/`.

### 3. Run the demo

Open `03_inference.ipynb` in Jupyter and run all cells in order.

**You will see the Gradio app appear inside the notebook cell** with a local URL:

- Running on local URL: `http://127.0.0.1:7860`

You have three options:

#### **Option A – Use inside notebook (default)**

The app runs inline inside the cell output. Paste your code directly there.

#### **Option B – Open in browser tab**

Copy the local URL (e.g., `http://127.0.0.1:7860`) and paste it into your browser while the cell is still running.

#### **Option C – Auto-open in browser**

Change the last line of the Gradio cell from:

```python
demo.launch(share=False)
```

to:

```python
demo.launch(share=False, inbrowser=True)
```

#### **Option D – Get a public shareable link (72 hours)**

Change the last line to:

```python
demo.launch(share=True)
```

This gives a public link like `https://abc123.gradio.live` that anyone can access without installing anything. Useful for showing to recruiters.

## Known Limitations

- Only corrects comments — not variable names, function names, or string literals
- Short words (under 4 characters) are skipped to avoid false positives
- Model sometimes generates contextually correct but spelling-incorrect words
  for patterns not well represented in training data
- Single typo correction per comment pass (multi-typo handled iteratively)

## Future Work

- Train on 50,000+ examples for better coverage of rare typo patterns
- Add Java and JavaScript support
- Build a VS Code extension for real-time comment correction
- Switch to T5-base or T5-large for better accuracy
- Auto-detect programming language from code syntax

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Model | T5-small (HuggingFace Transformers) |
| Training | PyTorch, Kaggle T4 GPU |
| Dataset | CodeSearchNet (HuggingFace Datasets) |
| Spell detection | pyspellchecker |
| Web demo | Gradio |
| Language | Python 3.12 |
| Environment | Anaconda, Jupyter Notebook |

## Development Journey

1. **Data Pipeline** — Loaded 412k Python functions from CodeSearchNet,
   filtered to 20k clean docstring lines, injected synthetic typos using
   4 strategies mimicking real human typing mistakes.

2. **Approach 1 (RoBERTa MLM)** — Fine-tuned RoBERTa on masked comment correction.
   Achieved 98% training accuracy but only 16.6% on unseen examples.
   Root cause: masking removes the typo so model ignores spelling entirely.

3. **Approach 2 (T5 seq2seq)** — Switched to T5 with input format
   `"fix: <corrupted comment>"`. Model sees the actual typo and rewrites
   the sentence correctly. Significant improvement on real examples.

4. **Inference Pipeline** — Built comment extractor supporting Python and C++,
   integrated T5 correction, rebuilt fixed code with changes tracked.

5. **Web Demo** — Gradio app where user pastes code, selects language,
   and gets corrected code with line-by-line change summary.

## Author

Shamili Yanamala

B.Tech Computer Science and Engineering

IIT Hyderabad
