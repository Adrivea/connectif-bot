"""
chat.py — Interfaz de chat interactiva para consultar la documentación de Connectif.

Carga el índice TF-IDF y permite hacer preguntas en lenguaje natural.
Devuelve los chunks más relevantes con título y URL del artículo.
"""

import os

import joblib
from sklearn.metrics.pairwise import cosine_similarity

INDEX_DIR = os.path.join(os.path.dirname(__file__), "data", "index")

SCORE_THRESHOLD = 0.1
TOP_K = 3


def load_index():
    """Carga el índice TF-IDF desde disco."""
    vectorizer = joblib.load(os.path.join(INDEX_DIR, "vectorizer.joblib"))
    tfidf_matrix = joblib.load(os.path.join(INDEX_DIR, "tfidf_matrix.joblib"))
    chunks = joblib.load(os.path.join(INDEX_DIR, "chunks.joblib"))
    return vectorizer, tfidf_matrix, chunks


def search(query, vectorizer, tfidf_matrix, chunks):
    """Busca los chunks más relevantes para una consulta."""
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Ordenar por score descendente
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    results = []
    seen_urls = set()
    for idx, score in ranked:
        if score < SCORE_THRESHOLD:
            break
        if len(results) >= TOP_K:
            break

        chunk = chunks[idx]
        # Evitar mostrar múltiples chunks del mismo artículo
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


def build_response(results):
    """Genera una respuesta en lenguaje natural a partir de los fragmentos encontrados."""
    lines = []

    # Cuerpo: texto combinado de los fragmentos, redactado como soporte
    if len(results) == 1:
        lines.append("Según la documentación de Connectif:\n")
    else:
        lines.append("Esto es lo que encontré en la documentación de Connectif:\n")

    for r in results:
        lines.append(f">> {r['title']}")
        # Mostrar el fragmento relevante de forma limpia
        preview = r["text"][:400].strip()
        if len(r["text"]) > 400:
            preview += "..."
        lines.append(f"  {preview}")
        lines.append("")

    # Sección de fuentes
    lines.append("Fuentes:")
    seen = set()
    for r in results:
        if r["url"] not in seen:
            seen.add(r["url"])
            lines.append(f"  - {r['title']}: {r['url']}")

    return "\n".join(lines)


def main():
    print("Cargando índice...")
    try:
        vectorizer, tfidf_matrix, chunks = load_index()
    except FileNotFoundError:
        print("Error: No se encontró el índice.")
        print("Ejecuta primero:")
        print("  python ingest.py")
        print("  python build_index.py")
        return

    print(f"Índice cargado: {len(chunks)} chunks de documentación.")
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
            print(build_response(results))
        else:
            print("\nNo encontré una respuesta clara sobre eso en la documentación.")
            print("Te recomiendo buscar directamente en el centro de ayuda:")
            print("  https://support.connectif.ai/hc/es")

        print()


if __name__ == "__main__":
    main()
