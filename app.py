"""
app.py — Asistente Connectif (Docs) — Interfaz web con Streamlit.

Reutiliza el indice TF-IDF de data/index/ y presenta respuestas
en lenguaje natural con tono de agente de soporte.
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
TOP_K = 5

HELP_EMAIL = os.environ.get("HELP_EMAIL", "")
HELP_WHATSAPP = os.environ.get("HELP_WHATSAPP", "")
HELP_FORM = "https://support.connectif.ai/hc/es/requests/new"

# ---------------------------------------------------------------------------
# Cargar indice (cacheado por Streamlit)
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
# Deteccion del tipo de pregunta
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
# Construccion de respuesta humana
# ---------------------------------------------------------------------------

def build_response(query, results):
    """Genera markdown con respuesta adaptada al tipo de pregunta."""
    intent = detect_intent(query)
    parts = []

    if intent == "how":
        parts.append("Estos son los pasos que encontre en la documentacion:\n")
        for i, r in enumerate(results, 1):
            preview = r["text"][:500].strip()
            if len(r["text"]) > 500:
                preview += "..."
            parts.append(f"**Paso {i} — {r['title']}**\n")
            parts.append(f"{preview}\n")

    elif intent == "what":
        parts.append("Segun la documentacion de Connectif:\n")
        main = results[0]
        preview = main["text"][:600].strip()
        if len(main["text"]) > 600:
            preview += "..."
        parts.append(f"**{main['title']}**\n")
        parts.append(f"{preview}\n")
        if len(results) > 1:
            parts.append("\n**Tambien puede interesarte:**\n")
            for r in results[1:]:
                parts.append(f"- {r['title']}\n")

    elif intent == "error":
        parts.append("Checklist de diagnostico basado en la documentacion:\n")
        for i, r in enumerate(results, 1):
            preview = r["text"][:400].strip()
            if len(r["text"]) > 400:
                preview += "..."
            parts.append(f"- [ ] **{r['title']}**: {preview}\n")

    else:
        parts.append("Esto es lo que encontre en la documentacion de Connectif:\n")
        for r in results:
            preview = r["text"][:450].strip()
            if len(r["text"]) > 450:
                preview += "..."
            parts.append(f"**{r['title']}**\n")
            parts.append(f"{preview}\n")

    return "\n".join(parts)

# ---------------------------------------------------------------------------
# Bloque de fuentes
# ---------------------------------------------------------------------------

def render_sources(results):
    st.markdown("---")
    st.markdown("**Fuentes:**")
    seen = set()
    for r in results:
        if r["url"] not in seen:
            seen.add(r["url"])
            st.markdown(f"- [{r['title']}]({r['url']})")

# ---------------------------------------------------------------------------
# Bloque de escalamiento
# ---------------------------------------------------------------------------

def render_escalation():
    st.markdown("---")
    st.markdown("**Mesa de ayuda**")
    lines = []
    if HELP_EMAIL:
        lines.append(f"- Email: **{HELP_EMAIL}**")
    if HELP_WHATSAPP:
        lines.append(f"- WhatsApp: **{HELP_WHATSAPP}**")
    lines.append(f"- [Enviar solicitud al soporte de Connectif]({HELP_FORM})")
    st.markdown("\n".join(lines))

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Asistente Connectif (Docs)",
        page_icon="📘",
        layout="centered",
    )

    st.title("Asistente Connectif (Docs)")
    st.caption("Busca en la documentacion oficial de Connectif")

    # Cargar indice
    try:
        vectorizer, tfidf_matrix, chunks = load_index()
    except FileNotFoundError:
        st.error(
            "No se encontro el indice. Ejecuta primero:\n\n"
            "```\npython ingest.py\npython build_index.py\n```"
        )
        return

    # Toggle diagnostico
    diag = st.toggle("Modo diagnostico", value=False)

    # Formulario de pregunta
    with st.form("ask", clear_on_submit=False):
        query = st.text_input(
            "Escribe tu pregunta:",
            placeholder="Ej: Como crear un workflow de carrito abandonado?",
        )
        submitted = st.form_submit_button("Preguntar")

    if not submitted or not query.strip():
        return

    query = query.strip()

    # Buscar
    results = search(query, vectorizer, tfidf_matrix, chunks)

    if results:
        # Respuesta principal
        response_md = build_response(query, results)
        st.markdown(response_md)

        # Boton copiar
        st.code(response_md, language=None)
        st.caption("Selecciona el texto del bloque de arriba para copiar la respuesta.")

        # Fuentes
        render_sources(results)

        # Diagnostico
        if diag:
            st.markdown("---")
            st.markdown("**Modo diagnostico — Fragments / Scores:**")
            for r in results:
                st.markdown(f"- `{r['score']:.4f}` — **{r['title']}**")
                with st.expander(f"Fragment: {r['title'][:50]}"):
                    st.text(r["text"])
    else:
        st.warning("No encontre una respuesta clara en la documentacion.")
        st.markdown(
            "**Intenta reformular tu pregunta:**\n"
            "- Usa palabras clave especificas (ej: *workflow*, *segmento*, *email*)\n"
            "- Describe la accion que quieres realizar (ej: *como crear...*, *como configurar...*)\n"
            "- Menciona la seccion de Connectif (ej: *editor de email*, *cupones*, *integracion*)"
        )
        render_escalation()


if __name__ == "__main__":
    main()
