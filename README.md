# FinanceInsight-NER

Financial Named Entity Recognition system for extracting important entities from financial text using Transformer-based token classification, BIO tagging, automatic annotation, and financial pattern rules.

## Overview

FinanceInsight-NER processes financial text and identifies entities such as organizations, people, locations, dates, money values, percentages, metrics, products, and financial events.

The project combines general NER predictions and financial regex rules to create initial annotations, converts them into BIO format, and trains a Transformer-based token classification model.

## Key Features

* Financial text preprocessing
* Financial dataset integration
* Automatic entity annotation
* BIO tagging
* Transformer-based token classification
* Financial entity extraction
* Entity-level evaluation
* JSON-based prediction output
* Command-line demonstration scripts

## Entity Types

| Entity  | Description       | Example     |
| ------- | ----------------- | ----------- |
| DATE    | Financial dates   | Q2 2024     |
| EVENT   | Financial events  | acquisition |
| LOC     | Locations         | New York    |
| METRIC  | Financial metrics | revenue     |
| MONEY   | Monetary values   | $25 billion |
| ORG     | Organizations     | Tesla       |
| PERCENT | Percentages       | 15%         |
| PERSON  | People            | Elon Musk   |
| PRODUCT | Products          | Model 3     |

## Dataset

The project uses financial data collected from multiple sources.

The merged dataset contains approximately 19,880 rows.

| Source           |   Rows |
| ---------------- | -----: |
| PDF text         | 10,719 |
| Clean text       |  5,524 |
| Tesla stock data |  3,637 |
| Total            | 19,880 |

Average sentence length is approximately 30.56 words, with an average of 35.25 tokens.

## Data Processing

The preprocessing pipeline:

Raw Financial Data
→ Text Cleaning
→ Financial Symbol Preservation
→ Sentence Processing
→ Dataset Preparation
→ Automatic Annotation
→ BIO Tagging
→ Model Training

Important financial information such as currency symbols, percentages, financial numbers, ticker symbols, and dates is preserved during preprocessing.

## Automatic Annotation

Initial annotations are generated using a hybrid approach:

* General NER model for entities such as organizations, people, and locations
* Regex rules for financial patterns such as money, percentages, dates, and ticker symbols

The generated annotations can then be reviewed and corrected before training.

## Model

The project uses Hugging Face Transformers for token classification.

Main components include:

* AutoTokenizer
* AutoModelForTokenClassification
* Trainer
* PyTorch
* Seqeval

The model predicts a BIO label for each token in the input text.

## BIO Tagging

BIO represents:

* B: Beginning of an entity
* I: Inside an entity
* O: Outside an entity

Example:

```text
Tesla       B-ORG
reported    O
revenue     B-METRIC
of          O
$25         B-MONEY
billion     I-MONEY
```

## Evaluation

The model is evaluated using precision, recall, and F1-score with Seqeval.

| Metric    | Score |
| --------- | ----: |
| Precision |  0.85 |
| Recall    |  0.89 |
| F1-score  |  0.87 |

The overall F1-score is approximately 87%.

### Entity-Level Performance

| Entity  | F1-score |
| ------- | -------: |
| DATE    |     0.93 |
| EVENT   |     1.00 |
| LOC     |     0.84 |
| METRIC  |     1.00 |
| MONEY   |     0.87 |
| ORG     |     0.39 |
| PERCENT |     1.00 |
| PRODUCT |     0.96 |

Organization recognition is the main area requiring improvement, with an F1-score of 0.39.

## Project Structure

```text
FinanceInsight-NER/
│
├── Demo_final_BIO.py
├── bio_data_label_analysis.py
├── clean_bio.py
├── cli_grouped_ner_demo.py
├── cli_user_filtered_ner_demo.py
├── document_level_ner_parser.py
├── finbert_ner_evaluation.py
├── final_document_output.json
├── document_level_ner_sample_output.txt
├── parsed_tables.json
├── trainf2.txt
├── testf2.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Sreekanth-Reddy1729/-FinanceInsight-NER.git
cd -FinanceInsight-NER
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Activate it on Linux or macOS:

```bash
source venv/bin/activate
```

Install the required libraries:

```bash
pip install transformers torch datasets pandas numpy seqeval scikit-learn
```

## Usage

Run the BIO demonstration:

```bash
python Demo_final_BIO.py
```

The demonstration loads the trained NER model and extracts financial entities from input text.

Example:

```text
Input:
Tesla reported revenue of $25.5 billion in Q2 2024.

Output:
ORG: Tesla
METRIC: revenue
MONEY: $25.5 billion
DATE: Q2 2024
```

## Technologies

Python
Hugging Face Transformers
PyTorch
Pandas
NumPy
Datasets
Seqeval
Regex
Label Studio
Git
GitHub

## Future Improvements

* Increase manually verified training data
* Improve organization recognition
* Improve annotation quality
* Balance entity classes
* Perform detailed error analysis
* Experiment with financial pretrained models
* Optimize training hyperparameters
* Deploy the model using FastAPI, Docker, or AWS

## Project Highlights

```text
Dataset Size       : 19,880 rows
Entity Classes     : 9
NER Format         : BIO
Model Type         : Transformer Token Classification
Precision          : 0.85
Recall             : 0.89
Overall F1         : 0.87
```

## Author

Sreekath Reddy

Computer Science Engineering
Lovely Professional University

## License

This project is intended for educational and research purposes.

