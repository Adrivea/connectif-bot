"""
app.py — Guia inteligente de Connectif

Flujo: Pregunta -> buscar en docs (RAG) -> GPT redacta respuesta humana -> mostrar
"""

import hashlib
import os
import re

import joblib
import openai
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

INDEX_DIR = os.path.join(os.path.dirname(__file__), "data", "index")
SCORE_THRESHOLD = 0.1
MAX_RESULTS = 5
MAX_CHUNK_CHARS = 1200

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

HELP_EMAIL = os.environ.get("HELP_EMAIL", "")
HELP_WHATSAPP = os.environ.get("HELP_WHATSAPP", "")

# ---------------------------------------------------------------------------
# System prompt para GPT
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
Eres un asistente experto en analisis documental basado EXCLUSIVAMENTE en la informacion recuperada desde un sistema RAG.

Tu funcion es:
- Explicar de forma clara, humana y estructurada el contenido de los documentos fuente.
- NO interpretar mas alla de lo que el texto permite.
- NO inventar requisitos, pasos, plazos, autoridades o conclusiones.

Recibiras informacion en forma de CHUNKS provenientes de documentos RAW originales.
Cada chunk corresponde a un fragmento literal de uno o varios documentos fuente.

Los documentos RAW son la fuente de verdad absoluta.
Los CHUNKS son recortes parciales de esos documentos.

- Solo puedes responder usando informacion que este explicitamente contenida en los chunks.
- Si un dato NO aparece en los chunks, debes decir claramente:
  "Esta informacion no se encuentra en los documentos consultados."
- Nunca completes vacios con conocimiento general, experiencia previa o suposiciones.
- No alucines.
- No inventes pasos, requisitos, plazos, costos, autoridades ni procedimientos.
- No mezcles conocimiento externo al RAG.
- No "mejores" el contenido con interpretaciones propias.
- No extrapoles normas, practicas comunes o supuestos.
- Si el documento es ambiguo o incompleto, indicalo.

ESTRUCTURA OBLIGATORIA DE RESPUESTA

Tu respuesta DEBE seguir EXACTAMENTE esta estructura:

1. Explicacion basada en los documentos
Explica en lenguaje humano, claro y comprensible.
Resume lo que realmente dicen los documentos.
Usa parrafos cortos.
No repitas texto literal innecesariamente.
No agregues informacion que no este en los chunks.
Ejemplo de tono: "Segun los documentos consultados, se establece que..."

2. **Checklist de acciones para el cliente** (SOLO si esta en los documentos)
Presenta un checklist claro y accionable, unicamente con acciones que esten explicitamente mencionadas o claramente descritas en los documentos.
Usa este formato EXACTO con casillas de verificacion:

**Checklist para el cliente:**
- [ ] Accion concreta 1
- [ ] Accion concreta 2
- [ ] Accion concreta 3

Si los documentos no permiten construir un checklist completo, indica:
"Los documentos no contienen informacion suficiente para definir un checklist completo de acciones."

3. Fuentes documentales
Incluye SIEMPRE un apartado de fuentes.
Formato obligatorio:
Fuentes:
- [Titulo del articulo](URL real)
Si hay multiples documentos, enumeralos.

MANEJO DE INCERTIDUMBRE (MUY IMPORTANTE)
Si la pregunta del usuario excede lo que dicen los documentos:
"La informacion solicitada no se encuentra en los documentos analizados. Para responder con certeza, seria necesario contar con documentacion adicional."
Esto NO es un error. Es el comportamiento correcto.

TONO Y ESTILO
- Lenguaje humano, profesional y claro.
- Nada de jerga tecnica innecesaria.
- Nada de tono academico pesado.
- Nada de frases tipo "en general", "normalmente", "usualmente".
- Precision > fluidez.

PRINCIPIO FINAL (EL MAS IMPORTANTE)
Prefiere una respuesta incompleta pero correcta
antes que una respuesta completa pero inventada.
""".strip()

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
.block-container {
    max-width: 960px !important;
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
.response-container {
    max-width: 900px;
    margin: 1rem auto;
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 14px;
    padding: 2rem 2.5rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    color: #1F2937;
    font-size: 0.95rem;
    line-height: 1.75;
    text-align: justify;
}
.response-container h2 {
    font-size: 1.35rem;
    font-weight: 700;
    color: #1E3A5F;
    margin: 0 0 1.2rem 0;
    padding-bottom: 0.6rem;
    border-bottom: 2px solid #2563EB;
    text-align: left;
}
.response-container h4 {
    font-size: 1.05rem;
    font-weight: 700;
    color: #1E3A5F;
    margin: 1.5rem 0 0.5rem 0;
    text-align: left;
}
.response-container p {
    margin: 0.5rem 0;
    text-align: justify;
}
.response-container ul, .response-container ol {
    padding-left: 1.5rem;
    margin: 0.5rem 0;
}
.response-container li {
    margin-bottom: 0.4rem;
    text-align: left;
}
.response-container a {
    color: #2563EB;
    text-decoration: none;
}
.response-container a:hover {
    text-decoration: underline;
}
.response-container .section-divider {
    border: none;
    border-top: 1px solid #E5E7EB;
    margin: 1.2rem 0;
}
.contact-footer {
    max-width: 900px;
    margin: 0.75rem auto 1.5rem auto;
    background: #EFF6FF;
    border: 1px solid #BFDBFE;
    border-radius: 10px;
    padding: 1rem 1.5rem;
    font-size: 0.9rem;
    color: #1E3A5F;
    text-align: center;
    line-height: 1.6;
}
.contact-footer strong {
    color: #1E3A5F;
}
.contact-footer a {
    color: #2563EB;
    text-decoration: none;
    font-weight: 600;
}
.contact-footer a:hover {
    text-decoration: underline;
}
.no-results-card {
    max-width: 900px;
    margin: 1rem auto;
    background: #FEF2F2;
    border: 1px solid #FECACA;
    border-radius: 12px;
    padding: 1rem 1.25rem;
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


def _generar_respuesta_gpt(query, results):
    """Envia chunks como contexto a GPT y devuelve respuesta en lenguaje humano."""
    # Construir contexto con los chunks
    context_parts = []
    for i, r in enumerate(results[:MAX_RESULTS], 1):
        text = r["text"][:MAX_CHUNK_CHARS]
        context_parts.append(
            f"--- Fuente {i}: {r['title']} ---\n"
            f"URL: {r['url']}\n"
            f"{text}"
        )
    context = "\n\n".join(context_parts)

    sources = "\n".join(
        f"- [{r['title']}]({r['url']})" for r in results[:5]
    )

    user_msg = (
        f"CONTEXTO (fragmentos de documentacion):\n\n{context}\n\n"
        f"FUENTES DISPONIBLES:\n{sources}\n\n"
        f"PREGUNTA DEL USUARIO:\n{query}"
    )

    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=1500,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )
    return response.choices[0].message.content


def _fallback_respuesta(query, results):
    """Respuesta sin LLM: muestra texto limpio del chunk."""
    principal = results[0]
    texto = _limpiar_texto(principal["text"])

    parts = []
    parts.append(f"*{query.strip()}*\n")
    if texto:
        parts.append(texto)
    else:
        parts.append(f"La documentacion tiene informacion en: *{principal['title']}*")
    parts.append("")
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


def _process_inline(text):
    """Convierte markdown inline (bold, links) a HTML."""
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(
        r'\[([^\]]+)\]\(([^)]+)\)',
        r'<a href="\2" target="_blank">\1</a>',
        text,
    )
    return text


def _md_to_html(md_text):
    """Convierte markdown basico de la respuesta GPT a HTML estructurado."""
    lines = md_text.split("\n")
    html_parts = []
    in_ul = False
    in_ol = False
    para_buf = []

    def flush_para():
        nonlocal para_buf
        if para_buf:
            html_parts.append(f'<p>{" ".join(para_buf)}</p>')
            para_buf = []

    def close_lists():
        nonlocal in_ul, in_ol
        if in_ul:
            html_parts.append("</ul>")
            in_ul = False
        if in_ol:
            html_parts.append("</ol>")
            in_ol = False

    for line in lines:
        stripped = line.strip()

        # Linea vacia
        if not stripped:
            flush_para()
            close_lists()
            continue

        # Encabezados
        if stripped.startswith("#### "):
            flush_para()
            close_lists()
            html_parts.append(f'<h4>{_process_inline(stripped[5:])}</h4>')
            continue
        if stripped.startswith("### "):
            flush_para()
            close_lists()
            html_parts.append(f'<h4>{_process_inline(stripped[4:])}</h4>')
            continue
        if stripped.startswith("## "):
            flush_para()
            close_lists()
            html_parts.append('<hr class="section-divider">')
            html_parts.append(f'<h4>{_process_inline(stripped[3:])}</h4>')
            continue
        if stripped.startswith("# "):
            flush_para()
            close_lists()
            html_parts.append(f'<h4>{_process_inline(stripped[2:])}</h4>')
            continue

        # Checkbox list (- [ ] item)
        m_checkbox = re.match(r'^[-*]\s*\[[ x]\]\s+(.+)', stripped)
        if m_checkbox:
            flush_para()
            if in_ol:
                html_parts.append("</ol>")
                in_ol = False
            if not in_ul:
                html_parts.append('<ul style="list-style:none;padding-left:0.5rem;">')
                in_ul = True
            html_parts.append(
                f'<li style="margin-bottom:0.5rem;">'
                f'\u2610 {_process_inline(m_checkbox.group(1))}</li>'
            )
            continue

        # Lista desordenada (- o * o checklist emoji)
        m_ul = re.match(r'^[-*]\s+(.+)', stripped)
        m_check = re.match(r'^[\u2610\u2611\u2612\u2713\u2714\u2716]\s*\d*\.?\s*(.+)', stripped)
        if m_ul or m_check:
            flush_para()
            if in_ol:
                html_parts.append("</ol>")
                in_ol = False
            if not in_ul:
                html_parts.append("<ul>")
                in_ul = True
            content = m_ul.group(1) if m_ul else m_check.group(1)
            html_parts.append(f'<li>{_process_inline(content)}</li>')
            continue

        # Lista ordenada (1. 2. etc.)
        m_ol = re.match(r'^(\d+)\.\s+(.+)', stripped)
        if m_ol:
            flush_para()
            if in_ul:
                html_parts.append("</ul>")
                in_ul = False
            if not in_ol:
                html_parts.append("<ol>")
                in_ol = True
            html_parts.append(f'<li>{_process_inline(m_ol.group(2))}</li>')
            continue

        # Texto normal -> parrafo
        para_buf.append(_process_inline(stripped))

    flush_para()
    close_lists()
    return "\n".join(html_parts)


def _mostrar_respuesta(query, vectorizer, tfidf_matrix, chunks):
    """Flujo: buscar (RAG) -> GPT redacta respuesta -> mostrar."""
    # Bubble del usuario
    st.markdown(
        f'<div class="bubble-user">{_html_escape(query)}</div>',
        unsafe_allow_html=True,
    )

    # Buscar en docs
    results = buscar(query, vectorizer, tfidf_matrix, chunks)

    # Si NO hay resultados
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

    # Generar respuesta humana con GPT (o fallback si no hay API key)
    cache_key = "r_" + hashlib.md5(query.strip().lower().encode()).hexdigest()
    if cache_key in st.session_state:
        md = st.session_state[cache_key]
    else:
        if OPENAI_API_KEY:
            try:
                with st.spinner("Generando respuesta..."):
                    md = _generar_respuesta_gpt(query, results)
            except Exception as e:
                st.warning(f"Error con OpenAI: {e}. Mostrando texto directo.")
                md = _fallback_respuesta(query, results)
        else:
            md = _fallback_respuesta(query, results)
        st.session_state[cache_key] = md

    # Convertir markdown a HTML y renderizar en contenedor profesional
    body_html = _md_to_html(md)
    full_html = (
        '<div class="response-container">'
        f'<h2>{_html_escape(query)}</h2>'
        f'{body_html}'
        '</div>'
    )
    st.markdown(full_html, unsafe_allow_html=True)

    # Footer de contacto
    contact_parts = []
    if HELP_EMAIL:
        contact_parts.append(
            f'<strong>Email:</strong> <a href="mailto:{HELP_EMAIL}">{HELP_EMAIL}</a>'
        )
    if HELP_WHATSAPP:
        contact_parts.append(
            f'<strong>WhatsApp:</strong> <a href="https://wa.me/{HELP_WHATSAPP}" target="_blank">{HELP_WHATSAPP}</a>'
        )
    contact_line = " &nbsp;|&nbsp; ".join(contact_parts) if contact_parts else ""
    footer_html = (
        '<div class="contact-footer">'
        'Si tienes preguntas adicionales, no dudes en contactar a nuestra '
        '<strong>mesa de ayuda</strong>.'
    )
    if contact_line:
        footer_html += f'<br>{contact_line}'
    footer_html += '</div>'
    st.markdown(footer_html, unsafe_allow_html=True)

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
