"""
ingest.py — Descarga artículos del help center de Connectif.

Usa la API pública de Zendesk Help Center para obtener artículos
en español y guardarlos como JSON.
"""

import json
import os
import re
import time
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")

BASE_URL = "https://support.connectif.ai"
API_URL = f"{BASE_URL}/api/v2/help_center"
LOCALE = "es"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}


def html_to_text(html):
    """Convierte HTML a texto plano limpio."""
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    return soup.get_text(separator="\n", strip=True)


def fetch_articles():
    """Obtiene todos los artículos en español via la API de Zendesk."""
    articles = []
    url = f"{API_URL}/{LOCALE}/articles.json?per_page=100"

    while url:
        print(f"  Solicitando: {url}")
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code != 200:
                print(f"  HTTP {resp.status_code}")
                # Si la API con locale falla, intentar sin locale
                if LOCALE in url and "/es/" in url:
                    url = f"{API_URL}/articles.json?per_page=100"
                    print(f"  Reintentando sin locale: {url}")
                    continue
                break

            data = resp.json()
            batch = data.get("articles", [])
            articles.extend(batch)
            print(f"  Recibidos {len(batch)} artículos (total: {len(articles)})")

            # Paginación
            url = data.get("next_page")

        except requests.RequestException as e:
            print(f"  Error: {e}")
            break

        time.sleep(1)

    return articles


def save_article(article):
    """Guarda un artículo de la API como JSON en data/raw/."""
    article_id = article.get("id", "unknown")
    title = article.get("title", "Sin título")
    body_html = article.get("body", "")
    text = html_to_text(body_html)

    if not text:
        return None

    # Construir URL legible del artículo
    url = article.get("html_url", f"{BASE_URL}/hc/es/articles/{article_id}")

    data = {
        "url": url,
        "title": title,
        "text": text,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }

    path = os.path.join(DATA_DIR, f"{article_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return path


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Descargando artículos via API de Zendesk Help Center...\n")

    # 1. Obtener artículos
    articles = fetch_articles()
    print(f"\nObtenidos {len(articles)} artículos de la API.\n")

    if not articles:
        print("No se obtuvieron artículos.")
        print("Verifica que el help center sea público en:")
        print(f"  {BASE_URL}/hc/es")
        return

    # 2. Guardar cada artículo
    saved = 0
    skipped = 0
    for i, article in enumerate(articles, 1):
        title = article.get("title", "?")
        path = save_article(article)
        if path:
            saved += 1
            text_len = len(html_to_text(article.get("body", "")))
            print(f"  [{i}/{len(articles)}] {title[:60]} ({text_len} chars)")
        else:
            skipped += 1
            print(f"  [{i}/{len(articles)}] {title[:60]} — sin contenido, saltado")

    print(f"\nDescarga completada: {saved} guardados, {skipped} saltados.")
    print(f"Directorio: {DATA_DIR}")


if __name__ == "__main__":
    main()
