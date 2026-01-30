"""
app.py — Guia inteligente de Connectif

Flujo: Pregunta -> buscar en docs -> si hay texto mostrarlo ordenado -> si no decir que no esta -> FIN
"""

import os
import re

import joblib
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

INDEX_DIR = os.path.join(os.path.dirname(__file__), "data", "index")
SCORE_THRESHOLD = 0.1
MAX_RESULTS = 5

HELP_EMAIL = os.environ.get("HELP_EMAIL", "")
HELP_WHATSAPP = os.environ.get("HELP_WHATSAPP", "")

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
.block-container {
    max-width: 760px !important;
    padding-top: 2rem !important;
}
.app-header {
    text-align: center;
    padding: 1.5rem 0 0.5rem 0;
}
.app-header h1 {
    font-size: 1.75rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 0.25rem;
}
.app-header p {
    font-size: 0.95rem;
    color: #6B7280;
    margin: 0;
}
.bubble-user {
    background: #2563EB;
    color: #FFFFFF;
    border-radius: 18px 18px 4px 18px;
    padding: 0.75rem 1.15rem;
    margin: 0.75rem 0;
    max-width: 80%;
    margin-left: auto;
    font-size: 0.95rem;
    line-height: 1.5;
}
.bubble-bot {
    background: #F3F4F6;
    color: #111827;
    border: 1px solid #E5E7EB;
    border-radius: 18px 18px 18px 4px;
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
    max-width: 95%;
    font-size: 0.93rem;
    line-height: 1.65;
}
.bubble-bot h3, .bubble-bot h4 {
    margin-top: 0.75rem;
    margin-bottom: 0.25rem;
    color: #111827;
}
.bubble-bot ul, .bubble-bot ol {
    padding-left: 1.25rem;
    margin: 0.35rem 0;
}
.bubble-bot li {
    margin-bottom: 0.2rem;
}
.no-results-card {
    background: #FEF2F2;
    border: 1px solid #FECACA;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin: 0.75rem 0;
}
.no-results-card h4 {
    color: #991B1B;
    margin: 0 0 0.5rem 0;
}
.no-results-card p, .no-results-card li {
    font-size: 0.9rem;
    color: #7F1D1D;
}
.faq-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #374151;
    margin-bottom: 0.5rem;
}
</style>
"""

# ---------------------------------------------------------------------------
# Preguntas frecuentes
# ---------------------------------------------------------------------------

FAQ_QUESTIONS = [
    "Como instalar Connectif en mi web?",
    "Como integrar Connectif con Shopify?",
    "Como integrar Connectif con Tiendanube?",
    "Como integrar Connectif con BigCommerce?",
    "Como saber si Connectif esta bien instalado?",
    "Como importar contactos a Connectif?",
    "Como crear un segmento?",
    "Que es un workflow y como se crea?",
    "Como crear una campana de email?",
    "Como funcionan las recomendaciones personalizadas?",
]

# ---------------------------------------------------------------------------
# Cargar indice
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Cargando documentacion...")
def load_index():
    vectorizer = joblib.load(os.path.join(INDEX_DIR, "vectorizer.joblib"))
    tfidf_matrix = joblib.load(os.path.join(INDEX_DIR, "tfidf_matrix.joblib"))
    chunks = joblib.load(os.path.join(INDEX_DIR, "chunks.joblib"))
    return vectorizer, tfidf_matrix, chunks

# ---------------------------------------------------------------------------
# Busqueda
# ---------------------------------------------------------------------------

def _merge_article_chunks(texts):
    """Une chunks del mismo articulo eliminando solapamiento."""
    if not texts:
        return ""
    if len(texts) == 1:
        return texts[0]
    merged = texts[0]
    for i in range(1, len(texts)):
        next_text = texts[i]
        overlap_window = min(120, len(merged), len(next_text))
        best_overlap = 0
        for ol in range(overlap_window, 10, -1):
            if merged[-ol:] == next_text[:ol]:
                best_overlap = ol
                break
        if best_overlap > 0:
            merged += next_text[best_overlap:]
        else:
            merged += "\n\n" + next_text
    return merged


def buscar(query, vectorizer, tfidf_matrix, chunks):
    """Busca en el indice y devuelve resultados agrupados por articulo."""
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    top_chunks = []
    for idx, score in ranked:
        if score < SCORE_THRESHOLD:
            break
        if len(top_chunks) >= 50:
            break
        chunk = chunks[idx]
        if len(chunk["text"].strip()) < 50:
            continue
        top_chunks.append((idx, score, chunk))

    # Agrupar por articulo, hasta 3 chunks por articulo
    articles = {}
    for idx, score, chunk in top_chunks:
        url = chunk["url"]
        if url not in articles:
            articles[url] = {
                "title": chunk["title"],
                "url": url,
                "best_score": score,
                "chunk_list": [],
            }
        if score > articles[url]["best_score"]:
            articles[url]["best_score"] = score
        if len(articles[url]["chunk_list"]) < 3:
            articles[url]["chunk_list"].append((idx, chunk["text"]))

    sorted_articles = sorted(
        articles.values(), key=lambda a: a["best_score"], reverse=True
    )[:MAX_RESULTS]

    results = []
    for art in sorted_articles:
        art["chunk_list"].sort(key=lambda c: c[0])
        merged = _merge_article_chunks([t for _, t in art["chunk_list"]])
        results.append({
            "score": art["best_score"],
            "title": art["title"],
            "url": art["url"],
            "text": merged,
        })
    return results

# ---------------------------------------------------------------------------
# Limpiar y ordenar texto para mostrar
# ---------------------------------------------------------------------------

def _limpiar_texto(text):
    """Limpia un texto de chunk para mostrarlo legible."""
    # Unir espacios y saltos excesivos
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Recortar si empieza a mitad de palabra
    if text and (text[0].islower() or text[0] in "\xe1\xe9\xed\xf3\xfa\xf1"):
        m = re.search(r"[.!?\n]\s+([A-Z\xc1\xc9\xcd\xd3\xda\xd1])", text)
        if m:
            text = text[m.end() - 1:]

    # Filtrar lineas basura
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if len(stripped) < 10:
            continue
        if stripped[0] in (",", ";", ")"):
            continue
        lines.append(stripped)

    return "\n".join(lines).strip()


def _formatear_respuesta(query, results):
    """Construye el markdown con el texto del chunk limpio y legible."""
    principal = results[0]
    texto = _limpiar_texto(principal["text"])

    parts = []

    # Titulo
    parts.append(f"*{query.strip()}*\n")

    # Contenido exacto del chunk
    if texto:
        parts.append(texto)
    else:
        parts.append(f"La documentacion tiene informacion en: *{principal['title']}*")
    parts.append("")

    # Fuentes (1-5)
    parts.append("**Fuentes:**\n")
    seen = set()
    for r in results[:5]:
        if r["url"] not in seen:
            seen.add(r["url"])
            parts.append(f"- [{r['title']}]({r['url']})")
    parts.append("")

    return "\n".join(parts)

# ---------------------------------------------------------------------------
# Renderizado
# ---------------------------------------------------------------------------

def _html_escape(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _mostrar_respuesta(query, vectorizer, tfidf_matrix, chunks):
    """Flujo completo: buscar -> mostrar texto del chunk -> o decir que no esta."""
    # Bubble del usuario
    st.markdown(
        f'<div class="bubble-user">{_html_escape(query)}</div>',
        unsafe_allow_html=True,
    )

    # Buscar
    results = buscar(query, vectorizer, tfidf_matrix, chunks)

    # Si NO hay texto -> decir que no esta en la doc
    if not results:
        st.markdown(
            '<div class="no-results-card">'
            "<h4>No esta en la documentacion.</h4>"
            "<p>No encontre informacion sobre esto en los articulos de Connectif. "
            "Intenta con otras palabras clave o busca directamente en "
            '<a href="https://support.connectif.ai/hc/es" target="_blank">'
            "support.connectif.ai</a>.</p></div>",
            unsafe_allow_html=True,
        )
        return

    # Si hay texto -> mostrarlo ordenado
    md = _formatear_respuesta(query, results)
    st.markdown(
        f'<div class="bubble-bot">\n\n{md}\n\n</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Guia inteligente de Connectif",
        page_icon=":blue_book:",
        layout="centered",
    )

    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown(
        '<div class="app-header">'
        "<h1>Guia inteligente de Connectif</h1>"
        "<p>Asistente de consulta con IA, basado en la documentacion oficial de Connectif.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    try:
        vectorizer, tfidf_matrix, chunks = load_index()
    except FileNotFoundError:
        st.error("No se encontro el indice. Ejecuta: python ingest.py && python build_index.py")
        return

    with st.form("ask", clear_on_submit=False):
        query = st.text_input(
            "Escribe tu pregunta:",
            placeholder="Ej: Como crear un workflow de carrito abandonado?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Preguntar", use_container_width=True)

    st.markdown('<p class="faq-title">Preguntas frecuentes</p>', unsafe_allow_html=True)
    faq_left = FAQ_QUESTIONS[: (len(FAQ_QUESTIONS) + 1) // 2]
    faq_right = FAQ_QUESTIONS[(len(FAQ_QUESTIONS) + 1) // 2:]
    col1, col2 = st.columns(2)
    faq_clicked = ""
    for q in faq_left:
        if col1.button(q, key=f"faq_{q}", use_container_width=True):
            faq_clicked = q
    for q in faq_right:
        if col2.button(q, key=f"faq_{q}", use_container_width=True):
            faq_clicked = q

    active_query = ""
    if faq_clicked:
        active_query = faq_clicked
    elif submitted and query.strip():
        active_query = query.strip()

    if not active_query:
        return

    _mostrar_respuesta(active_query, vectorizer, tfidf_matrix, chunks)


if __name__ == "__main__":
    main()
