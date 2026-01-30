"""
build_index.py — Construye el índice TF-IDF a partir de los artículos descargados.

Lee los JSON de data/raw/, divide el texto en chunks, genera una
matriz TF-IDF y serializa todo con joblib para uso posterior.
"""

import json
import os
import re

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "data", "index")

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_articles():
    """Carga todos los artículos JSON desde data/raw/."""
    articles = []
    for filename in os.listdir(RAW_DIR):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(RAW_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            articles.append(data)
    return articles


def clean_text(text):
    """Limpia texto: quita espacios múltiples y líneas vacías excesivas."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Divide texto en chunks de ~size caracteres con overlap."""
    if len(text) <= size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap

    return [c for c in chunks if c]


def build_chunks(articles):
    """Crea chunks con metadatos a partir de todos los artículos."""
    all_chunks = []
    for article in articles:
        text = clean_text(article.get("text", ""))
        if not text:
            continue

        title = article.get("title", "Sin título")
        url = article.get("url", "")

        chunks = chunk_text(text)
        for chunk in chunks:
            all_chunks.append({
                "text": chunk,
                "title": title,
                "url": url,
            })

    return all_chunks


def main():
    os.makedirs(INDEX_DIR, exist_ok=True)

    # 1. Cargar artículos
    articles = load_articles()
    print(f"Artículos cargados: {len(articles)}")

    if not articles:
        print("No hay artículos en data/raw/. Ejecuta ingest.py primero.")
        return

    # 2. Crear chunks
    chunks = build_chunks(articles)
    print(f"Chunks generados: {len(chunks)}")

    if not chunks:
        print("No se generaron chunks. Verifica los artículos descargados.")
        return

    # 3. Construir TF-IDF
    texts = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        stop_words=None,  # Mantener stopwords en español para mejor matching
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"Matriz TF-IDF: {tfidf_matrix.shape}")

    # 4. Guardar índice
    joblib.dump(vectorizer, os.path.join(INDEX_DIR, "vectorizer.joblib"))
    joblib.dump(tfidf_matrix, os.path.join(INDEX_DIR, "tfidf_matrix.joblib"))
    joblib.dump(chunks, os.path.join(INDEX_DIR, "chunks.joblib"))

    print(f"\nÍndice guardado en {INDEX_DIR}/")
    print(f"  vectorizer.joblib")
    print(f"  tfidf_matrix.joblib")
    print(f"  chunks.joblib")


if __name__ == "__main__":
    main()
