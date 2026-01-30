"""
chat.py — Interfaz de chat CLI para consultar la documentacion de Connectif.

Carga el indice TF-IDF y permite hacer preguntas en lenguaje natural.
Genera respuestas con formato humano, fuentes y mesa de ayuda.
"""

import os
import re

import joblib
from sklearn.metrics.pairwise import cosine_similarity

INDEX_DIR = os.path.join(os.path.dirname(__file__), "data", "index")

SCORE_THRESHOLD = 0.1
TOP_K = 5

HELP_EMAIL = os.environ.get("HELP_EMAIL", "")
HELP_WHATSAPP = os.environ.get("HELP_WHATSAPP", "")
HELP_FORM = "https://support.connectif.ai/hc/es/requests/new"

# ---------------------------------------------------------------------------
# Deteccion de intent
# ---------------------------------------------------------------------------

_HOW_PATTERN = re.compile(
    r"\b(como|cómo|pasos|paso a paso|configurar|crear|hacer|activar|integrar|implementar)\b", re.I
)
_WHAT_PATTERN = re.compile(
    r"\b(que es|qué es|que son|qué son|para que sirve|para qué sirve|definici[oó]n)\b", re.I
)
_ERROR_PATTERN = re.compile(
    r"\b(error|problema|no funciona|falla|fallo|no aparece|no llega|no se ve|no carga)\b", re.I
)


def detect_intent(query):
    if _ERROR_PATTERN.search(query):
        return "error"
    if _HOW_PATTERN.search(query):
        return "how"
    if _WHAT_PATTERN.search(query):
        return "what"
    return "general"

# ---------------------------------------------------------------------------
# Indice
# ---------------------------------------------------------------------------

def load_index():
    vectorizer = joblib.load(os.path.join(INDEX_DIR, "vectorizer.joblib"))
    tfidf_matrix = joblib.load(os.path.join(INDEX_DIR, "tfidf_matrix.joblib"))
    chunks = joblib.load(os.path.join(INDEX_DIR, "chunks.joblib"))
    return vectorizer, tfidf_matrix, chunks


def search(query, vectorizer, tfidf_matrix, chunks):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    results = []
    seen_urls = set()
    for idx, score in ranked:
        if score < SCORE_THRESHOLD:
            break
        if len(results) >= TOP_K:
            break
        chunk = chunks[idx]
        if chunk["url"] in seen_urls:
            continue
        seen_urls.add(chunk["url"])
        results.append({
            "score": float(score),
            "title": chunk["title"],
            "url": chunk["url"],
            "text": chunk["text"],
        })
    return results

# ---------------------------------------------------------------------------
# Fusion y deduplicacion
# ---------------------------------------------------------------------------

def _clean_fragment(text):
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _deduplicate_sentences(fragments):
    seen = set()
    unique_lines = []
    for frag in fragments:
        clean = _clean_fragment(frag)
        for line in clean.split("\n"):
            line = line.strip()
            if not line:
                continue
            key = re.sub(r"[^\w\s]", "", line.lower()).strip()
            if len(key) < 15:
                continue
            if key not in seen:
                seen.add(key)
                unique_lines.append(line)
    return unique_lines

# ---------------------------------------------------------------------------
# Respuesta formateada
# ---------------------------------------------------------------------------

def build_response(query, results):
    intent = detect_intent(query)
    all_fragments = [r["text"] for r in results]
    unique_lines = _deduplicate_sentences(all_fragments)
    summarized = unique_lines[:12]
    titles = list(dict.fromkeys(r["title"] for r in results))

    parts = []

    # Titulo
    parts.append(f"  {query.strip().rstrip('?')}?")
    parts.append("  " + "=" * 50)
    parts.append("")

    # Explicacion
    if intent == "how":
        parts.append(f"  En la documentacion de Connectif encontre informacion relevante")
        parts.append(f"  sobre este tema, principalmente en: {titles[0]}.")
        if len(titles) > 1:
            otros = ", ".join(titles[1:3])
            parts.append(f"  Tambien hay info relacionada en: {otros}.")
        parts.append("")
        parts.append("  Pasos para hacerlo:")
        parts.append("")
        for i, line in enumerate(summarized[:8], 1):
            parts.append(f"    {i}. {line}")
    elif intent == "what":
        parts.append(f"  Segun la documentacion oficial de Connectif,")
        parts.append(f"  este tema se cubre en el articulo: {titles[0]}.")
        parts.append("")
        parts.append("  Resumen:")
        parts.append("")
        for line in summarized[:6]:
            parts.append(f"    - {line}")
    elif intent == "error":
        parts.append("  Entiendo que estas teniendo un problema.")
        parts.append("  Revise la documentacion y encontre estos puntos de diagnostico:")
        parts.append("")
        parts.append("  Checklist de diagnostico:")
        parts.append("")
        for line in summarized[:8]:
            parts.append(f"    [ ] {line}")
    else:
        parts.append(f"  Encontre informacion relevante en la documentacion de Connectif.")
        if titles:
            parts.append(f"  Articulos mas relacionados: {titles[0]}"
                         + (f" y {titles[1]}." if len(titles) > 1 else "."))
        parts.append("")
        for line in summarized[:8]:
            parts.append(f"    - {line}")

    parts.append("")

    # Que tener en cuenta
    if len(summarized) > 8:
        parts.append("  Que debes tener en cuenta:")
        parts.append("")
        for line in summarized[8:12]:
            parts.append(f"    - {line}")
        parts.append("")

    # Fuentes
    parts.append("  " + "-" * 50)
    parts.append("  Fuentes:")
    seen = set()
    for r in results[:5]:
        if r["url"] not in seen:
            seen.add(r["url"])
            parts.append(f"    - {r['title']}: {r['url']}")
    parts.append("")

    # Mesa de ayuda
    best_score = results[0]["score"] if results else 0
    if best_score < 0.25 or intent == "error":
        parts.append("  " + "-" * 50)
        parts.append("  Mesa de ayuda:")
        parts.append("  Si necesitas mas ayuda, contacta al soporte de Connectif:")
        if HELP_EMAIL:
            parts.append(f"    Email: {HELP_EMAIL}")
        if HELP_WHATSAPP:
            parts.append(f"    WhatsApp: {HELP_WHATSAPP}")
        parts.append(f"    Formulario: {HELP_FORM}")
        parts.append("")

    return "\n".join(parts)


def build_no_results_response(query):
    parts = []
    parts.append(f"  {query.strip().rstrip('?')}?")
    parts.append("  " + "=" * 50)
    parts.append("")
    parts.append("  No encontre esta informacion en la documentacion oficial de Connectif.")
    parts.append("")
    parts.append("  Intenta reformular tu pregunta:")
    parts.append("    - Usa palabras clave especificas (workflow, segmento, email)")
    parts.append("    - Describe la accion que quieres realizar (como crear..., como configurar...)")
    parts.append("    - Menciona la seccion de Connectif (editor de email, cupones, integracion)")
    parts.append("")
    parts.append("  " + "-" * 50)
    parts.append("  Mesa de ayuda:")
    parts.append("  Si necesitas mas ayuda, contacta al soporte de Connectif:")
    if HELP_EMAIL:
        parts.append(f"    Email: {HELP_EMAIL}")
    if HELP_WHATSAPP:
        parts.append(f"    WhatsApp: {HELP_WHATSAPP}")
    parts.append(f"    Formulario: {HELP_FORM}")
    return "\n".join(parts)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Cargando indice...")
    try:
        vectorizer, tfidf_matrix, chunks = load_index()
    except FileNotFoundError:
        print("Error: No se encontro el indice.")
        print("Ejecuta primero:")
        print("  python ingest.py")
        print("  python build_index.py")
        return

    print(f"Indice cargado: {len(chunks)} chunks de documentacion.")
    print("Escribe tu pregunta (o 'salir' para terminar).\n")

    while True:
        try:
            query = input("Pregunta: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego.")
            break

        if not query:
            continue
        if query.lower() in ("salir", "exit", "quit", "q"):
            print("Hasta luego.")
            break

        results = search(query, vectorizer, tfidf_matrix, chunks)

        if results:
            print()
            print(build_response(query, results))
        else:
            print()
            print(build_no_results_response(query))

        print()


if __name__ == "__main__":
    main()
