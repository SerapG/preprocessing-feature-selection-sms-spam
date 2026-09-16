# Preprocessing and Feature Selection for SMS Spam Classification

Code for an ablation study on SMS spam classification. The experiment grid
crosses **4 preprocessing pipelines** (basic, stopword removal, stemming,
full) × **2 feature selection methods** (Gini index, Distinguishing Feature
Selector / DFS) × **6 vocabulary sizes** (500, 300, 100, 50, 30, 10) ×
**5 classifiers** (SVM, Multinomial Naive Bayes, Random Forest, TextCNN,
LSTM) × **2 languages** (English, Turkish), for a total of **480
experiments**. Each experiment is scored with **Macro-F1** on a held-out
30% stratified test split (seed = 42).

## Setup

```bash
pip install -r requirements.txt
```

## Datasets

The datasets are **not included** in this repository — download them
separately and place them at the repository root:

- **English**: [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection).
  Extract the archive so the file `SMSSpamCollection` ends up at
  `English_sms_spam/SMSSpamCollection`.
- **Turkish**: the TurkishSMS collection introduced in A. K. Uysal,
  S. Gunal, S. Ergin, and E. S. Gunal, "A novel framework for SMS spam
  filtering," *INISTA 2012*, doi:10.1109/INISTA.2012.6246947. This dataset
  is not redistributed here; obtain it from the original authors and place
  `sms_legitimate.txt` and `sms_spam.txt` under `TurkishSMS/`.

### Expected folder structure

```
preprocessing-feature-selection-sms-spam/
├── English_sms_spam/
│   └── SMSSpamCollection
├── TurkishSMS/
│   ├── sms_legitimate.txt
│   └── sms_spam.txt
├── src/
├── results/
│   └── paper/
├── run.py
└── requirements.txt
```

`src/config.py` locates these two folders automatically by scanning
upward from its own location until it finds a directory containing both
`English_sms_spam/` and `TurkishSMS/` — so the layout above (datasets at
the repository root) works out of the box, and no path configuration is
required.

## Running

```bash
python run.py            # full 480-experiment grid (~26 minutes on CPU)
python run.py --smoke    # quick smoke test (~1 minute)
```

Each run writes a numbered folder under `results/`, containing:

- `final_results.csv` — long-format results (one row per experiment)
- `tables/` — 16 pivot tables (one per language × pipeline × method,
  rows = vocabulary size, columns = algorithm, values = Macro-F1)

## Paper tables

`results/paper/` contains the results reported in the paper: `final_results.csv`
holds all 480 Macro-F1 scores, and the files under `results/paper/tables/`
correspond directly to the Macro-F1 tables in the paper.
