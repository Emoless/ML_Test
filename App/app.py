# flask_app.py — Обновлённое Flask-приложение с универсальной фильтрацией текста
from flask import Flask, request, jsonify, render_template
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from bs4 import BeautifulSoup
import requests
import re

app = Flask(__name__)

# Загрузка модели и токенизатора
MODEL_PATH = "./furniture_ner_model"
model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/extract", methods=["POST"])
def extract():
    url = request.form.get("url")
    try:
        headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Удаление явного мусора
        for tag in soup(["script", "style", "footer", "nav", "header", "noscript"]):
            tag.decompose()

        # Получаем чистый текст
        text = soup.get_text(separator=" ", strip=True)

        # Убираем повторяющиеся заголовки и промо-тексты
        text = re.sub(r'(FREE SHIPPING[^.!?\n]+[.!?\n])', '', text, flags=re.IGNORECASE)
        text = re.sub(r'(\b[A-Z]{2,}(?:\s+[A-Z]{2,})+\b)', '', text)  # CAPS TEXT SPAM
        text = re.sub(r'(\d+\s*x\s*\d+\s*(cm|inches)?)', '', text, flags=re.IGNORECASE)  # габариты

        # Удаление повторов и сжатие пробелов
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(\b\w+\b)( \1\b)+', r'\1', text)  # повторяющиеся слова подряд

        # Усечём до разумного размера (например 4000 символов)
        clean_text = text[:4000]

        # Разбиение на фрагменты для модели
        chunks = [clean_text[i:i+512] for i in range(0, len(clean_text), 512)]

        # Обработка моделью
        raw_products = []
        for chunk in chunks:
            ents = ner_pipeline(chunk)
            for e in ents:
                if e.get("entity_group") == "PRODUCT":
                    word = e.get("word", "").replace("##", "").strip(" ,.-")
                    if len(word) > 2 and any(c.isalpha() for c in word):
                        raw_products.append(word)

        # Финальная фильтрация
        products = sorted(set(
            p for p in raw_products
            if p[0].isupper() or len(p) > 4
        ))

        return render_template(
            "index.html",
            url=url,
            products=products,
            extracted_text=clean_text
        )

    except Exception as e:
        return render_template("index.html", url=url, error=str(e))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
