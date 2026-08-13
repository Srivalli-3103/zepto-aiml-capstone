# Data Pipeline Submodule Documentation

This pipeline automates live web scraping, cleans raw text elements, generates structured relational database systems using an SQLite environment, and verifies relational output parity between standard query engines and pandas memory spaces.

## Installation & Environment Activation

To isolate execution layers specifically for this extraction runner, deploy a clean localized virtual environment:

```bash
cd data_pipeline
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r ../requirements.txt
```

## Production Pipeline Execution

Trigger the data collection and compliance mapping engine using the following terminal call:

```bash
python pipeline.py
```

## Engineering Design Choices & Data Rules

### 1. Constant Value Conversions
As outlined in the core business parameters, all currency adjustments use a fixed-rate project constant: **1 GBP = 105.50 INR**. This baseline configuration excludes temporal parameters or active time tracking, generating static pricing metrics (`price_inr`) at the extraction level.

### 2. Sanitization Pipeline & Strict Typings
* **Title Extraction:** Captured explicitly from raw HTML `title` attributes instead of visible text blocks to circumvent clipping induced by embedded ellipses (`...`).
* **Price Formats:** Cleans strings of alpha-character components (`£` and broken encoding artifacts like `Â`) prior to running numeric transformations to generate true numerical float typings (`REAL`).
* **Star Metrics:** Alphanumeric star designations (`One` through `Five`) are parsed directly via conversion dictionaries to integer ranges between `1` and `5`.
* **Stock Metrics:** Alphanumeric indicators are translated into simple bits (`0` or `1`) for clean compatibility with SQLite boolean handling.

### 3. Relational Schema Architecture
The SQLite instance enforces strict referential constraints (`PRAGMA foreign_keys = ON;`) to protect schema health:
* `categories`: Master classification lookup tracking unique alphanumeric configurations (`category_id` acts as Primary Key).
* `books`: Data repository containing descriptive indexes, clean floats, numeric scales, and relational ties (`category_id` acts as Foreign Key referencing `categories`).
