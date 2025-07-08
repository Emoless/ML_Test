# Furniture Product Extractor (ML_Test)

> **Test Assignment**: extract furniture product names from arbitrary e‑commerce pages using a fine‑tuned NER model, wrapped in a Flask web app.

---

## 🎯 Project Overview

This repository contains a proof‑of‑concept end‑to‑end solution:

1. **Data pipelines** (`RawData/`, `DataSets/`, `Scripts/`) to collect and preprocess training material.  
2. **Model training** (Transformers) to build a custom NER that recognizes `PRODUCT` entities.  
3. **Web deployment** (`App/`) — a Flask application that:
   - Scrapes any furniture‑store URL
   - Cleans and chunks HTML text
   - Runs the NER pipeline
   - Presents extracted product names in a simple UI


---

## 🚀 Features & Tech Stack

- **NER fine‑tuning**  
  – Custom `PRODUCT` tag on real furniture pages  
  – Transformer architecture (DistilBERT)

- **Data Collection & Preparation**  
  – Scraping scripts to build a high‑quality training set (`Scripts/`)  
  – Raw HTML snapshots (`RawData/`) and processed datasets (`DataSets/`)

- **Flask Web Service** (`App/`)  
  – Auto‑download of model (Dropbox link)  
  – Universal HTML cleanup (scripts, styles, headers, repeat‑filtering)  
  – Chunked inference for long pages  
  – Post‑processing: subtoken cleanup, case/length filtering, deduplication  

- **Key Libraries**  
  – `transformers`, `torch`  
  – `beautifulsoup4`, `requests`  
  – `Flask`, `gunicorn`  

---

## 📁 Repository Structure

```

ML\_Test/
├── App/                   # Flask application & templates
│   ├── app.py             # Main Flask service
│   └── templates/
│       └── index.html     # Input form & results UI
├── DataSets/              # Processed train/test datasets (JSON/BIO)
├── RawData/               # Raw HTML or sample pages for annotation
├── Scripts/               # ETL / scraping / annotation helper scripts
├── requirements.txt       # Python dependencies
└── README.md              # This overview

````

---

## ⚙️ Installation & Run Locally

1. **Clone** the repo  
   ```bash
   git clone https://github.com/Emoless/ML_Test.git
   cd ML_Test/App
   ```

2. **Clone** the model  
   Download the folder with model from https://www.dropbox.com/scl/fi/uk2r2zkuazid28nn5rv55/furniture_ner_model.zip?rlkey=d1sq69rj2xzterfaiaaxr92cn&st=hyjj5v9v&dl=0
   and place it in App folder (same folder with app.py). You will have this structure:
   ```

    ML\_Test/
    ├── App/                   # Flask application & templates
    │   ├── app.py             # Main Flask service
    │   ├── furniture_ner_model/...  #All needed files for model             
    │   └── templates/
    │       └── index.html     # Input form & results UI
    ├── DataSets/              # Processed train/test datasets (JSON/BIO)
    ├── RawData/               # Raw HTML or sample pages for annotation
    ├── Scripts/               # ETL / scraping / annotation helper scripts
    ├── requirements.txt       # Python dependencies
    └── README.md              # This overview

    ```


4. **Install** dependencies (no cache to save space)

   ```bash
   pip install --no-cache-dir -r ../requirements.txt
   ```

5. **Run** the service

   ```bash
   python app.py
   ```

6. **Open** your browser

   ```
   http://localhost:5000
   ```

---

## 📝 Usage

1. **Paste** any furniture‑store product URL.
2. **Click** “Extract Products”.
3. **View**:

   * Detected product names
   * Snippet of cleaned text that fed the model

Use this to validate model performance on real pages during your interview demo.

---


## 🔄 Extending & Evaluation

* **Add more scraping scripts** in `Scripts/` to enlarge your training set.
* **Experiment** with different Transformer backbones (e.g., `bert-base-uncased`, `roberta`).
* **Monitor** precision/recall/F1 on held‑out `DataSets/` to justify model choice.

---


Thank you for reviewing my solution!
*— Matthew ML & NLP Engineer Candidate*
