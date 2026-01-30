"""
app.py — Guia inteligente de Connectif — Interfaz web con Streamlit.

Reutiliza el indice TF-IDF de data/index/ y presenta respuestas
en lenguaje natural con tono de agente de soporte.
"""

import os
import re

import joblib
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# System prompt (referencia interna de comportamiento)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
Eres el "Asistente Connectif (Docs)", un asistente de ayuda para clientes que
responde EXCLUSIVAMENTE usando la documentacion oficial publica de Connectif.

REGLAS OBLIGATORIAS:
1. NUNCA copies ni pegues texto literal largo de la documentacion.
2. SIEMPRE debes LEER la informacion encontrada y REDACTAR una respuesta clara,
   humana y explicativa.
3. Si varios fragmentos dicen lo mismo, FUSIONA la informacion y ELIMINA
   repeticiones.
4. NO muestres bloques de texto crudo ni fragmentos duplicados.
5. NO inventes informacion.
6. Si no hay suficiente evidencia documental, responde:
   "No encontre esta informacion en la documentacion oficial de Connectif".
7. Usa un tono humano, cercano y orientado a clientes.
8. Explica paso a paso cuando aplique.
9. Usa listas, numeracion y subtitulos para facilitar la lectura.
10. Al FINAL muestra una seccion "Fuentes" con enlaces (min 1, max 5).
11. Si la pregunta no puede resolverse completamente, incluye "Mesa de ayuda".
""".strip()

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
# CSS embebido
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
/* ---- Layout general ---- */
.block-container {
    max-width: 760px !important;
    padding-top: 2rem !important;
}

/* ---- Header ---- */
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

/* ---- Card contenedora de pregunta ---- */
.question-card {
    background: #F9FAFB;
    border: 1px solid #E5E7EB;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}

/* ---- Chat bubbles ---- */
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
.bubble-bot h3 {
    font-size: 1.05rem;
    margin-top: 0.5rem;
    margin-bottom: 0.35rem;
    color: #111827;
}
.bubble-bot h4 {
    font-size: 0.95rem;
    margin-top: 0.75rem;
    margin-bottom: 0.25rem;
    color: #374151;
}
.bubble-bot ul, .bubble-bot ol {
    padding-left: 1.25rem;
    margin: 0.35rem 0;
}
.bubble-bot li {
    margin-bottom: 0.2rem;
}
.bubble-bot hr {
    border: none;
    border-top: 1px solid #D1D5DB;
    margin: 0.85rem 0;
}

/* ---- Source chips ---- */
.source-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 0.5rem;
}
.source-chip {
    display: inline-block;
    background: #EFF6FF;
    color: #1D4ED8;
    border: 1px solid #BFDBFE;
    border-radius: 20px;
    padding: 0.3rem 0.85rem;
    font-size: 0.82rem;
    text-decoration: none;
    transition: background 0.15s;
}
.source-chip:hover {
    background: #DBEAFE;
    text-decoration: none;
    color: #1E40AF;
}

/* ---- Mesa de ayuda card ---- */
.help-desk-card {
    background: #FEF3C7;
    border: 1px solid #FDE68A;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-top: 0.75rem;
}
.help-desk-card h4 {
    color: #92400E;
    margin: 0 0 0.5rem 0;
    font-size: 0.95rem;
}
.help-desk-card p, .help-desk-card a {
    font-size: 0.88rem;
    color: #78350F;
    margin: 0.15rem 0;
}
.help-desk-card a {
    color: #B45309;
    text-decoration: underline;
}

/* ---- No results card ---- */
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

/* ---- Diagnostico ---- */
.diag-label {
    font-size: 0.8rem;
    color: #6B7280;
    margin-top: 0.25rem;
}

/* ---- FAQ section ---- */
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
    "Que es el analisis RFM en Connectif?",
]

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
# Limpieza y fusion de fragmentos
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


def _summarize_lines(lines, max_lines=12):
    return lines[:max_lines]

# ---------------------------------------------------------------------------
# Construccion de respuesta (markdown para dentro de la burbuja)
# ---------------------------------------------------------------------------

def build_response_md(query, results):
    """Genera markdown de la respuesta (sin fuentes ni mesa de ayuda, esos se renderizan aparte)."""
    intent = detect_intent(query)
    all_fragments = [r["text"] for r in results]
    unique_lines = _deduplicate_sentences(all_fragments)
    summarized = _summarize_lines(unique_lines)
    titles = list(dict.fromkeys(r["title"] for r in results))

    parts = []

    # Titulo
    parts.append(f"### {query.strip().rstrip('?')}?\n")

    # Explicacion
    if intent == "how":
        parts.append(
            f"En la documentacion de Connectif encontre informacion relevante "
            f"sobre este tema, principalmente en: **{titles[0]}**."
        )
        if len(titles) > 1:
            otros = ", ".join(f"*{t}*" for t in titles[1:3])
            parts.append(f"Tambien hay informacion relacionada en: {otros}.")
        parts.append("")
        parts.append("A continuacion te explico los puntos clave:\n")
    elif intent == "what":
        parts.append(
            f"Segun la documentacion oficial de Connectif, "
            f"este tema se cubre en el articulo **{titles[0]}**."
        )
        parts.append("")
        parts.append("Aqui tienes un resumen de lo mas importante:\n")
    elif intent == "error":
        parts.append(
            "Entiendo que estas teniendo un problema. "
            "Revise la documentacion y encontre los siguientes puntos "
            "que pueden ayudarte a diagnosticarlo:"
        )
        parts.append("")
    else:
        parts.append(
            "Encontre informacion relevante sobre tu consulta "
            "en la documentacion de Connectif."
        )
        if titles:
            parts.append(
                f"Los articulos mas relacionados son: **{titles[0]}**"
                + (f" y *{titles[1]}*." if len(titles) > 1 else ".")
            )
        parts.append("")

    # Contenido principal
    if intent == "how" and summarized:
        parts.append("#### Pasos para hacerlo:\n")
        for i, line in enumerate(summarized[:8], 1):
            parts.append(f"{i}. {line}")
        parts.append("")
    elif intent == "what" and summarized:
        for line in summarized[:6]:
            parts.append(f"- {line}")
        parts.append("")
    elif intent == "error" and summarized:
        parts.append("#### Checklist de diagnostico:\n")
        for line in summarized[:8]:
            parts.append(f"- [ ] {line}")
        parts.append("")
    else:
        for line in summarized[:8]:
            parts.append(f"- {line}")
        parts.append("")

    # Que tener en cuenta
    if len(summarized) > 8:
        parts.append("#### Que debes tener en cuenta:\n")
        for line in summarized[8:12]:
            parts.append(f"- {line}")
        parts.append("")

    # Checklist rapido (solo how)
    if intent == "how" and titles:
        parts.append("#### Checklist rapido:\n")
        for t in titles[:3]:
            parts.append(f"- [ ] Revisa el articulo: {t}")
        parts.append("- [ ] Verifica que los cambios se guardaron correctamente")
        parts.append("- [ ] Prueba el resultado en tu cuenta de Connectif")
        parts.append("")

    return "\n".join(parts)

# ---------------------------------------------------------------------------
# Renderizado HTML
# ---------------------------------------------------------------------------

def _html_escape(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_user_bubble(query):
    st.markdown(
        f'<div class="bubble-user">{_html_escape(query)}</div>',
        unsafe_allow_html=True,
    )


def render_bot_bubble(md_content):
    st.markdown(
        f'<div class="bubble-bot">\n\n{md_content}\n\n</div>',
        unsafe_allow_html=True,
    )


def render_source_chips(results):
    chips_html = '<div class="source-chips">'
    seen = set()
    for r in results[:5]:
        if r["url"] not in seen:
            seen.add(r["url"])
            title_esc = _html_escape(r["title"])
            chips_html += (
                f'<a class="source-chip" href="{r["url"]}" target="_blank">'
                f'{title_esc}</a>'
            )
    chips_html += "</div>"
    st.markdown(chips_html, unsafe_allow_html=True)


def render_help_desk():
    lines = []
    if HELP_EMAIL:
        lines.append(f"<p>Email: <strong>{_html_escape(HELP_EMAIL)}</strong></p>")
    if HELP_WHATSAPP:
        lines.append(f"<p>WhatsApp: <strong>{_html_escape(HELP_WHATSAPP)}</strong></p>")
    lines.append(
        f'<p><a href="{HELP_FORM}" target="_blank">'
        f"Enviar solicitud al soporte de Connectif</a></p>"
    )
    body = "\n".join(lines)
    st.markdown(
        f'<div class="help-desk-card">'
        f"<h4>Mesa de ayuda</h4>"
        f"<p>Si necesitas mas ayuda, contacta al equipo de soporte:</p>"
        f"{body}</div>",
        unsafe_allow_html=True,
    )


def render_no_results(query):
    st.markdown(
        f'<div class="no-results-card">'
        f"<h4>No encontre esta informacion en la documentacion oficial de Connectif.</h4>"
        f"<p>Intenta reformular tu pregunta:</p>"
        f"<ul>"
        f"<li>Usa palabras clave especificas (workflow, segmento, email)</li>"
        f"<li>Describe la accion que quieres realizar (como crear..., como configurar...)</li>"
        f"<li>Menciona la seccion de Connectif (editor de email, cupones, integracion)</li>"
        f"</ul></div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# UI principal
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Guia inteligente de Connectif",
        page_icon=":blue_book:",
        layout="centered",
    )

    # Inicializar session_state
    if "faq_query" not in st.session_state:
        st.session_state.faq_query = ""

    # CSS global
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Header
    st.markdown(
        '<div class="app-header">'
        "<h1>Guia inteligente de Connectif</h1>"
        "<p>Consulta la documentacion oficial de Connectif de forma rapida y clara.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Cargar indice
    try:
        vectorizer, tfidf_matrix, chunks = load_index()
    except FileNotFoundError:
        st.error(
            "No se encontro el indice. Ejecuta primero:\n\n"
            "```\npython ingest.py\npython build_index.py\n```"
        )
        return

    # --- Preguntas frecuentes ---
    st.markdown(
        '<p class="faq-title">Preguntas frecuentes</p>',
        unsafe_allow_html=True,
    )
    # Dos columnas
    faq_left = FAQ_QUESTIONS[: (len(FAQ_QUESTIONS) + 1) // 2]
    faq_right = FAQ_QUESTIONS[(len(FAQ_QUESTIONS) + 1) // 2:]
    col1, col2 = st.columns(2)
    for q in faq_left:
        if col1.button(q, key=f"faq_{q}", use_container_width=True):
            st.session_state.faq_query = q
    for q in faq_right:
        if col2.button(q, key=f"faq_{q}", use_container_width=True):
            st.session_state.faq_query = q

    # Controles: toggle diagnostico
    diag = st.toggle("Modo diagnostico", value=False)

    # Card de pregunta
    st.markdown('<div class="question-card">', unsafe_allow_html=True)
    with st.form("ask", clear_on_submit=False):
        query = st.text_input(
            "Escribe tu pregunta:",
            value=st.session_state.faq_query,
            placeholder="Ej: Como crear un workflow de carrito abandonado?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Preguntar", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Limpiar faq_query despues de usarla para evitar persistencia
    if st.session_state.faq_query:
        st.session_state.faq_query = ""

    if not submitted or not query.strip():
        return

    query = query.strip()

    # Burbuja del usuario
    render_user_bubble(query)

    # Buscar
    results = search(query, vectorizer, tfidf_matrix, chunks)

    if results:
        # Burbuja del bot con respuesta formateada
        response_md = build_response_md(query, results)
        render_bot_bubble(response_md)

        # Fuentes como chips
        st.markdown(
            '<p style="font-size:0.85rem;color:#374151;margin-top:0.75rem;">'
            "<strong>Fuentes:</strong></p>",
            unsafe_allow_html=True,
        )
        render_source_chips(results)

        # Mesa de ayuda si evidencia debil o error
        best_score = results[0]["score"] if results else 0
        intent = detect_intent(query)
        if best_score < 0.25 or intent == "error":
            render_help_desk()

        # Copiar respuesta
        with st.expander("Copiar respuesta"):
            st.code(response_md, language="markdown")

        # Diagnostico
        if diag:
            st.markdown(
                '<p class="diag-label"><strong>Modo diagnostico — Fragments / Scores:</strong></p>',
                unsafe_allow_html=True,
            )
            for r in results:
                st.markdown(f"- `{r['score']:.4f}` — **{r['title']}**")
                with st.expander(f"Fragment: {r['title'][:50]}"):
                    st.text(r["text"])
    else:
        render_no_results(query)
        render_help_desk()


if __name__ == "__main__":
    main()
