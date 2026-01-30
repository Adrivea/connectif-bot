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

# Ruta del logo (opcional, puede no existir)
LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
SHOW_LOGO = os.path.exists(LOGO_PATH)

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

1. (SIN TITULO de seccion, empieza directamente con la explicacion)
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

IMPORTANTE: NO incluyas seccion de fuentes ni enlaces a documentos. Las fuentes se muestran automaticamente por el sistema.

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
# CSS Premium - Estilo guía premium
# ---------------------------------------------------------------------------

PREMIUM_CSS = """
<style>
/* Ocultar elementos por defecto de Streamlit */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Fondo suave y elegante */
.stApp {
    background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
}

/* Contenedor principal centrado con max-width 900px */
.main-container {
    max-width: 900px;
    margin: 0 auto;
    padding: 2rem 1.5rem;
    box-sizing: border-box;
    width: 100%;
}

/* Forzar max-width en el contenedor de Streamlit */
.block-container {
    max-width: 900px !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    box-sizing: border-box !important;
}

/* Asegurar que el contenido esté centrado */
.stApp > div {
    max-width: 100%;
}

section[data-testid="stSidebar"] {
    background-color: #ffffff;
}

/* Ocultar el contenedor del formulario que crea Streamlit */
form[data-testid="stForm"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

div[data-testid="stForm"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Header premium con logo en esquina superior derecha */
.app-header {
    position: relative;
    text-align: center;
    padding: 2.5rem 0 2rem 0;
    margin-bottom: 2rem;
    min-height: 80px;
}

.app-header .logo-container {
    position: absolute;
    top: 0;
    right: 0;
    z-index: 10;
    padding: 0.5rem;
}

.app-header .logo-container img {
    max-height: 60px;
    max-width: 120px;
    width: auto;
    height: auto;
    display: block;
    object-fit: contain;
}

.app-header h1 {
    font-size: 2.25rem;
    font-weight: 700;
    color: #1a202c;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.02em;
}

.app-header p {
    font-size: 1.1rem;
    color: #64748b;
    margin: 0;
    font-weight: 400;
    line-height: 1.6;
}

/* Input principal elegante (sin card) */
.stTextInput > div > div > input {
    font-size: 1rem;
    padding: 0.875rem 1.25rem;
    border-radius: 12px;
    border: 2px solid #e2e8f0;
    transition: all 0.2s;
    margin-bottom: 1rem;
}

.stTextInput > div > div > input:focus {
    border-color: #3b82f6;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}

/* Botón llamativo */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
    color: white;
    font-weight: 600;
    font-size: 1rem;
    padding: 0.875rem 2rem;
    border-radius: 12px;
    border: none;
    transition: all 0.2s;
    box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.3);
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
    box-shadow: 0 6px 12px -1px rgba(59, 130, 246, 0.4);
    transform: translateY(-1px);
}

/* Sección FAQ con chips/botones en grid - Azul claro */
.faq-section {
    max-width: 900px;
    margin: 2rem auto;
    padding: 1.5rem;
    background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
    border: 1px solid #7dd3fc;
    border-radius: 12px;
    box-sizing: border-box;
}

.faq-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #0c4a6e;
    margin-bottom: 1rem;
    text-align: center;
}

.faq-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 0.75rem;
    margin-top: 1rem;
}

.faq-chip {
    background: #ffffff;
    border: 2px solid #e2e8f0;
    border-radius: 12px;
    padding: 0.875rem 1.25rem;
    font-size: 0.95rem;
    color: #334155;
    cursor: pointer;
    transition: all 0.2s;
    text-align: center;
    font-weight: 500;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}

.faq-chip:hover {
    border-color: #3b82f6;
    background: #eff6ff;
    color: #1e40af;
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(59, 130, 246, 0.15);
}

/* Card de respuesta premium */
.response-card {
    background: #ffffff;
    border-radius: 16px;
    padding: 2.5rem;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    margin: 2rem 0;
    border: 1px solid rgba(226, 232, 240, 0.8);
    max-width: 900px;
    margin-left: auto;
    margin-right: auto;
}

.response-card h2 {
    font-size: 1.75rem;
    font-weight: 700;
    color: #1a202c;
    margin: 0 0 1.5rem 0;
    padding-bottom: 1rem;
    border-bottom: 3px solid #3b82f6;
    line-height: 1.3;
}

.response-card h4 {
    font-size: 1.2rem;
    font-weight: 700;
    color: #1e293b;
    margin: 2rem 0 0.75rem 0;
    line-height: 1.4;
}

.response-card p {
    font-size: 1rem;
    line-height: 1.8;
    color: #334155;
    margin: 0.75rem 0;
    text-align: justify;
}

.response-card ul, .response-card ol {
    padding-left: 1.75rem;
    margin: 1rem 0;
}

.response-card li {
    font-size: 1rem;
    line-height: 1.8;
    color: #334155;
    margin-bottom: 0.5rem;
    text-align: left;
}

.response-card a {
    color: #3b82f6;
    text-decoration: none;
    font-weight: 500;
}

.response-card a:hover {
    text-decoration: underline;
}

/* Checklist estilizado */
.response-card ul[style*="list-style:none"] {
    padding-left: 0;
}

.response-card ul[style*="list-style:none"] li {
    padding-left: 1.5rem;
    position: relative;
}

/* Sección de fuentes elegante */
.sources-section {
    max-width: 900px;
    margin: 1.5rem auto;
    padding: 0 1.5rem;
    box-sizing: border-box;
    width: 100%;
}

.sources-section h4 {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1e293b;
    margin: 0 0 1rem 0;
}

.source-card {
    display: flex;
    align-items: flex-start;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    text-decoration: none;
    color: #1f2937;
    transition: all 0.2s;
    max-width: 100%;
    word-wrap: break-word;
    overflow-wrap: break-word;
}

.source-card:hover {
    border-color: #3b82f6;
    background: #eff6ff;
    box-shadow: 0 4px 8px rgba(59, 130, 246, 0.1);
    text-decoration: none;
    transform: translateX(4px);
}

.source-card .source-icon {
    font-size: 1.5rem;
    margin-right: 1rem;
    flex-shrink: 0;
    opacity: 0.7;
    margin-top: 0.125rem;
}

.source-card > div {
    flex: 1;
    min-width: 0;
    overflow: hidden;
}

.source-card .source-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #1e293b;
    line-height: 1.4;
    word-wrap: break-word;
    overflow-wrap: break-word;
    hyphens: auto;
    margin: 0;
}

.source-card .source-url {
    font-size: 0.85rem;
    color: #64748b;
    margin-top: 0.25rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    word-break: break-all;
}

/* Footer de contacto */
.contact-footer {
    max-width: 900px;
    margin: 1.5rem auto;
    background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
    border: 1px solid #bfdbfe;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    font-size: 0.95rem;
    color: #1e3a5f;
    text-align: center;
    line-height: 1.7;
}

.contact-footer a {
    color: #2563eb;
    text-decoration: none;
    font-weight: 600;
}

.contact-footer a:hover {
    text-decoration: underline;
}

/* Card de no resultados */
.no-results-card {
    max-width: 900px;
    margin: 2rem auto;
    background: #fef2f2;
    border: 1px solid #fecaca;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
}

.no-results-card h4 {
    color: #991b1b;
    margin: 0 0 0.75rem 0;
    font-size: 1.2rem;
}

.no-results-card p {
    font-size: 1rem;
    color: #7f1d1d;
    line-height: 1.6;
}

/* Modo diagnóstico (solo visible si está activado) */
.diagnostic-section {
    max-width: 900px;
    margin: 1.5rem auto;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.5rem;
}

.diagnostic-section h4 {
    font-size: 1.1rem;
    font-weight: 600;
    color: #1e293b;
    margin: 0 0 1rem 0;
}

.diagnostic-item {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.9rem;
    color: #475569;
}

.diagnostic-item strong {
    color: #1e293b;
}

/* Separador de secciones */
.section-divider {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 1.5rem 0;
}

/* Ajustes de espaciado general */
.stMarkdown {
    margin-bottom: 0;
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
# Busqueda (MOTOR RAG - NO TOCAR)
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
    # Validar inputs
    if not query or not query.strip():
        return []
    
    if vectorizer is None or tfidf_matrix is None or chunks is None:
        raise ValueError("El índice no está cargado correctamente")
    
    try:
        query_vec = vectorizer.transform([query])
        scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    except Exception as e:
        raise ValueError(f"Error al procesar la consulta: {str(e)}")

    top_chunks = []
    try:
        for idx, score in ranked:
            if score < SCORE_THRESHOLD:
                break
            if len(top_chunks) >= 50:
                break
            if idx >= len(chunks):
                continue
            chunk = chunks[idx]
            if not isinstance(chunk, dict) or "text" not in chunk:
                continue
            if len(chunk["text"].strip()) < 50:
                continue
            top_chunks.append((idx, score, chunk))
    except (IndexError, KeyError, TypeError) as e:
        raise ValueError(f"Error al procesar los chunks: {str(e)}")

    # Agrupar por articulo, hasta 3 chunks por articulo
    articles = {}
    try:
        for idx, score, chunk in top_chunks:
            if not isinstance(chunk, dict) or "url" not in chunk or "title" not in chunk:
                continue
            url = chunk["url"]
            if url not in articles:
                articles[url] = {
                    "title": chunk.get("title", "Sin título"),
                    "url": url,
                    "best_score": score,
                    "chunk_list": [],
                }
            if score > articles[url]["best_score"]:
                articles[url]["best_score"] = score
            if len(articles[url]["chunk_list"]) < 3:
                articles[url]["chunk_list"].append((idx, chunk.get("text", "")))
    except Exception as e:
        raise ValueError(f"Error al agrupar resultados: {str(e)}")

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
# Renderizado de respuesta premium
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


def _mostrar_respuesta(query, vectorizer, tfidf_matrix, chunks, show_diagnostic=False):
    """Flujo: buscar (RAG) -> GPT redacta respuesta -> mostrar."""
    # Buscar en docs con manejo de errores
    try:
        results = buscar(query, vectorizer, tfidf_matrix, chunks)
    except Exception as e:
        st.error(f"Error al buscar en la documentación: {str(e)}")
        st.markdown(
            '<div class="no-results-card">'
            "<h4>Error en la búsqueda.</h4>"
            "<p>Ocurrió un error al buscar en la documentación. Por favor, intenta de nuevo o contacta a "
            '<a href="https://support.connectif.ai/hc/es/requests/new" target="_blank">'
            "nuestra mesa de ayuda</a>.</p></div>",
            unsafe_allow_html=True,
        )
        return

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

    # Modo diagnóstico (solo si está activado y hay datos)
    if show_diagnostic and results:
        diagnostic_html = '<div class="diagnostic-section"><h4>Modo Diagnóstico</h4>'
        diagnostic_html += f'<div class="diagnostic-item"><strong>Resultados encontrados:</strong> {len(results)}</div>'
        for i, r in enumerate(results[:3], 1):
            diagnostic_html += (
                f'<div class="diagnostic-item">'
                f'<strong>Fuente {i}:</strong> {_html_escape(r["title"])}<br>'
                f'<strong>Score:</strong> {r["score"]:.3f}<br>'
                f'<strong>URL:</strong> <a href="{r["url"]}" target="_blank">{_html_escape(r["url"])}</a>'
                f'</div>'
            )
        diagnostic_html += '</div>'
        st.markdown(diagnostic_html, unsafe_allow_html=True)

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

    # Convertir markdown a HTML y renderizar en card premium
    body_html = _md_to_html(md)
    full_html = (
        '<div class="response-card">'
        f'<h2>{_html_escape(query)}</h2>'
        f'{body_html}'
        '</div>'
    )
    st.markdown(full_html, unsafe_allow_html=True)

    # Fuentes como cajitas elegantes
    seen_urls = set()
    source_cards = []
    for r in results[:5]:
        if r["url"] in seen_urls:
            continue
        seen_urls.add(r["url"])
        title_esc = _html_escape(r["title"])
        url_short = r["url"].replace("https://", "").replace("http://", "")
        source_cards.append(
            f'<a class="source-card" href="{r["url"]}" target="_blank">'
            f'<span class="source-icon">📄</span>'
            f'<div>'
            f'<div class="source-title">{title_esc}</div>'
            f'<div class="source-url">{_html_escape(url_short)}</div>'
            f'</div></a>'
        )
    if source_cards:
        sources_html = (
            '<div class="sources-section">'
            '<h4>Documentacion consultada</h4>'
            + "".join(source_cards)
            + '</div>'
        )
        st.markdown(sources_html, unsafe_allow_html=True)

    # Footer de contacto
    email_line = ""
    if HELP_EMAIL:
        email_line = (
            f'<br><strong>Email:</strong> '
            f'<a href="mailto:{HELP_EMAIL}">{HELP_EMAIL}</a>'
        )
    footer_html = (
        '<div class="contact-footer">'
        'Si tienes preguntas adicionales, no dudes en contactar a nuestra '
        '<a href="https://support.connectif.ai/hc/es/requests/new" target="_blank">'
        'mesa de ayuda</a>.'
        f'{email_line}'
        '</div>'
    )
    st.markdown(footer_html, unsafe_allow_html=True)
    
    # Sección FAQ con chips en grid (después de las fuentes, en azul claro)
    _mostrar_faq()

# ---------------------------------------------------------------------------
# Mostrar FAQ
# ---------------------------------------------------------------------------

def _mostrar_faq():
    """Muestra la sección de preguntas frecuentes en azul claro."""
    st.markdown('<div class="faq-section">', unsafe_allow_html=True)
    st.markdown('<p class="faq-title">Preguntas frecuentes</p>', unsafe_allow_html=True)
    
    # Crear grid de FAQ usando columnas
    num_cols = 3
    cols = st.columns(num_cols)
    
    for idx, question in enumerate(FAQ_QUESTIONS):
        col_idx = idx % num_cols
        with cols[col_idx]:
            if st.button(question, key=f"faq_{idx}", use_container_width=True):
                # Al hacer clic: copiar pregunta al input y disparar búsqueda automáticamente
                st.session_state.faq_clicked = question
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Helper para logo
# ---------------------------------------------------------------------------

def _get_logo_base64():
    """Convierte el logo a base64 para mostrarlo en HTML."""
    import base64
    if SHOW_LOGO and os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return ""

# ---------------------------------------------------------------------------
# Main - UI Premium
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Guia inteligente de Connectif",
        page_icon=":blue_book:",
        layout="centered",
    )

    # Inyectar CSS premium
    st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

    # Contenedor principal centrado
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Header premium con logo (si existe), título y subtítulo
    header_html = '<div class="app-header">'
    if SHOW_LOGO:
        header_html += f'<div class="logo-container"><img src="data:image/png;base64,{_get_logo_base64()}" alt="Connectif Logo"></div>'
    header_html += (
        '<h1>Guia inteligente de Connectif</h1>'
        '<p>Asistente de consulta con IA, basado en la documentacion oficial de Connectif.</p>'
        '</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)

    # Cargar índice
    try:
        vectorizer, tfidf_matrix, chunks = load_index()
    except FileNotFoundError:
        st.error("No se encontro el indice. Ejecuta: python ingest.py && python build_index.py")
        return

    # Inicializar session_state para FAQ
    if "faq_clicked" not in st.session_state:
        st.session_state.faq_clicked = None

    # Formulario de pregunta (sin card blanco)
    with st.form("ask", clear_on_submit=False):
        # Si hay una pregunta del FAQ, usarla como valor inicial
        initial_value = st.session_state.faq_clicked if st.session_state.faq_clicked else ""
        
        query = st.text_input(
            "Escribe tu pregunta:",
            value=initial_value,
            placeholder="Ej: Como crear un workflow de carrito abandonado?",
            label_visibility="collapsed",
            key="query_input_field"
        )
        
        submitted = st.form_submit_button("Preguntar", use_container_width=True)

    # Modo diagnóstico (opcional, solo aparece si está activado)
    show_diagnostic = st.sidebar.checkbox("Modo diagnóstico", value=False, help="Muestra información técnica sobre la búsqueda")

    # Determinar query activa
    # Prioridad: FAQ clickeado > formulario enviado
    active_query = ""
    if st.session_state.faq_clicked:
        # Si hay FAQ clickeado, usarlo directamente (búsqueda automática)
        active_query = st.session_state.faq_clicked
        st.session_state.faq_clicked = None  # Resetear después de usar
    elif submitted and query.strip():
        active_query = query.strip()

    # Mostrar FAQ debajo del input si NO hay respuesta activa
    if not active_query:
        _mostrar_faq()

    # Mostrar respuesta si hay query activa
    if active_query:
        _mostrar_respuesta(active_query, vectorizer, tfidf_matrix, chunks, show_diagnostic)
    
    # Cerrar contenedor principal
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
