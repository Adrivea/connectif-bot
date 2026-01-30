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
Eres "Guia Inteligente de Connectif", un asistente de soporte que responde en espanol usando UNICAMENTE la documentacion oficial de Connectif que el sistema te entrega como CONTEXTO (extractos/fragmentos).

REGLAS DE ORO (OBLIGATORIAS)
1) NO inventes. NO asumas. NO uses conocimiento externo. Si algo no esta en el contexto, dilo con claridad.
2) Responde SIEMPRE con una explicacion util basada en el contexto. No redirijas a "consulta la guia" como respuesta principal.
3) NO dupliques contenido. Genera una sola respuesta final.
4) Formato limpio: nada de bloques repetidos, nada de listas pegadas, nada de emojis repetidos por todo el texto.
5) SIEMPRE incluye una seccion final llamada "Fuentes" con 1 a 5 enlaces (URLs) relevantes tomados del contexto.
6) Si la informacion del contexto es insuficiente o ambigua, incluye al final una seccion "Mesa de ayuda" con el email o WhatsApp configurado por el sistema, indicando que pueden ayudar con el caso.

ESTILO Y TONO
- Humano, claro, paciente, orientado a cliente.
- Explica como si la persona no fuera tecnica.
- Prioriza pasos accionables y definiciones simples.
- Usa frases cortas y estructura con subtitulos.
- No uses jerga innecesaria.

FORMATO DE SALIDA (ESTRUCTURA FIJA)
1) Titulo en cursiva con la pregunta del usuario:
   *<pregunta del usuario>*

2) "Respuesta"
   - Explica la idea principal en 2-5 lineas, directamente basada en el contexto.

3) "Como hacerlo" (SOLO si el contexto describe pasos o procedimiento)
   - Lista numerada 1., 2., 3... con pasos completos y ordenados.
   - Cada paso debe ser una accion concreta.
   - Si el contexto tiene requisitos previos, ponlos antes como "Antes de empezar".

4) "Notas importantes" (SOLO si el contexto menciona advertencias, limites, permisos, requisitos, compatibilidades)
   - Bullets cortos.

5) "Fuentes"
   - Lista con 1 a 5 links. Cada link en una linea.
   - No mas de 5.
   - No inventes URLs: solo las del contexto.

6) "Mesa de ayuda" (CONDICIONAL)
   Incluyela SOLO si:
   - el contexto no permite responder bien, o
   - hay multiples interpretaciones, o
   - el usuario necesita un dato que no esta en el contexto (p.ej. configuracion especifica de su cuenta).
   Debe decir:
   "Si necesitas que revisemos tu caso exacto, escribe a: <email> o WhatsApp: <whatsapp>"

MANEJO DE CONTEXTO
- Usa el contexto como fuente. Si el contexto contiene fragmentos irrelevantes, ignoralos.
- Si el contexto trae varias guias, elige la mas relevante y usa las otras como "Relacionado" SOLO si aportan (maximo 3 referencias, sin inflar la respuesta).
- Nunca pegues texto largo literal de la guia. Parafrasea y resume sin omitir informacion importante.

CHECK DE CALIDAD ANTES DE RESPONDER
- Estoy respondiendo la pregunta o solo mandando a leer?
- Estoy usando SOLO lo que esta en el contexto?
- Inclui "Fuentes" con 1 a 5 URLs?
- Evite duplicacion y listas raras?
- Use pasos numerados solo si realmente aplica?
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

def _merge_article_chunks(texts):
    """Une varios chunks del mismo articulo eliminando texto solapado."""
    if not texts:
        return ""
    if len(texts) == 1:
        return texts[0]
    merged = texts[0]
    for i in range(1, len(texts)):
        next_text = texts[i]
        # Intentar detectar solapamiento (los chunks tienen ~50 chars de overlap)
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


def search(query, vectorizer, tfidf_matrix, chunks):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    # Fase 1: recoger los mejores chunks (hasta 50) por encima del umbral
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

    # Fase 2: agrupar por articulo, hasta 3 chunks por articulo
    article_data = {}
    for idx, score, chunk in top_chunks:
        url = chunk["url"]
        if url not in article_data:
            article_data[url] = {
                "title": chunk["title"],
                "url": url,
                "best_score": score,
                "chunk_list": [],
            }
        if score > article_data[url]["best_score"]:
            article_data[url]["best_score"] = score
        if len(article_data[url]["chunk_list"]) < 3:
            article_data[url]["chunk_list"].append((idx, chunk["text"]))

    # Fase 3: ordenar articulos por mejor score, tomar top K
    sorted_articles = sorted(
        article_data.values(), key=lambda a: a["best_score"], reverse=True
    )[:TOP_K]

    # Fase 4: para cada articulo, unir chunks en orden de documento
    results = []
    for art in sorted_articles:
        art["chunk_list"].sort(key=lambda c: c[0])
        merged_text = _merge_article_chunks([text for _, text in art["chunk_list"]])
        results.append({
            "score": art["best_score"],
            "title": art["title"],
            "url": art["url"],
            "text": merged_text,
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
# Limpieza y normalizacion de texto
# ---------------------------------------------------------------------------

# Palabras que indican que la linea siguiente es continuacion de la anterior
_CONTINUATION = re.compile(r"^(en|y|de|para|con|que|a|al|del|la|el|los|las|un|una|o)\b", re.I)

# Patrones que indican pasos REALES en la documentacion
_REAL_STEP = re.compile(r"^\s*(\d+)\.\s+(.+)$")
_NAMED_STEP = re.compile(r"^\s*(Paso|PASO|Step)\s+\d+[.:]\s*(.+)$", re.I)


def _trim_broken_start(text):
    """Recorta texto que empieza a mitad de palabra u oracion."""
    if not text:
        return text
    first_char = text[0]
    # Si empieza con minuscula o caracter acentuado en minuscula -> fragmento roto
    if first_char.islower() or first_char in "\xe1\xe9\xed\xf3\xfa\xe0\xe8\xec\xf2\xf9\xf1":
        # Buscar el inicio de la primera oracion completa
        m = re.search(r"[.!?\n]\s+([A-Z\xc1\xc9\xcd\xd3\xda\xd1])", text)
        if m:
            return text[m.end() - 1:]
        # Si no hay oracion, buscar primer doble salto de linea
        nl = text.find("\n\n")
        if nl != -1 and nl < len(text) // 2:
            return text[nl + 2:]
        # Si nada funciona, quitar hasta el primer espacio (palabra partida)
        sp = text.find(" ")
        if sp != -1 and sp < 30:
            return text[sp + 1:]
    return text


def _normalize_text(text):
    """Normaliza texto: une lineas partidas, quita espacios repetidos."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Unir lineas partidas: si la linea no termina en punto/signo y la
    # siguiente empieza con palabra de continuacion o minuscula
    raw_lines = text.split("\n")
    merged = []
    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            merged.append("")
            continue
        if (merged
                and merged[-1]
                and not merged[-1].endswith((".", ":", "!", "?", ";"))
                and (_CONTINUATION.match(stripped) or stripped[0].islower())):
            merged[-1] = merged[-1] + " " + stripped
        else:
            merged.append(stripped)

    return "\n".join(merged)


def _extract_clean_text(fragment):
    """Limpia y normaliza un fragmento de texto."""
    text = _normalize_text(fragment)
    # Deduplicar oraciones y filtrar lineas rotas/muy cortas
    seen = set()
    unique = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            if unique and unique[-1] != "":
                unique.append("")
            continue
        # Descartar lineas muy cortas o que parecen fragmentos rotos
        if len(line) < 20:
            continue
        # Descartar lineas que empiezan con coma, punto y coma, o parentesis suelto
        if line[0] in (",", ";", ")"):
            continue
        key = re.sub(r"[^\w\s]", "", line.lower()).strip()
        if len(key) < 15:
            continue
        if key not in seen:
            seen.add(key)
            unique.append(line)
    return unique


def _has_real_steps(lines):
    """Detecta si el texto contiene pasos reales (numerados correctamente)."""
    step_count = 0
    for line in lines:
        if _REAL_STEP.match(line) or _NAMED_STEP.match(line):
            step_count += 1
    return step_count >= 2


def _extract_steps_and_prose(lines):
    """Separa lineas de pasos reales del resto (prosa)."""
    steps = []
    prose = []
    for line in lines:
        m = _REAL_STEP.match(line)
        m2 = _NAMED_STEP.match(line)
        if m:
            steps.append(m.group(2).strip())
        elif m2:
            steps.append(m2.group(2).strip())
        else:
            prose.append(line)
    return steps, prose

# ---------------------------------------------------------------------------
# Construccion de respuesta
# ---------------------------------------------------------------------------

def _clean_chunk_for_display(text):
    """Prepara el texto de un chunk para mostrarlo como respuesta legible."""
    text = _trim_broken_start(text)
    text = _normalize_text(text)
    # Quitar lineas muy cortas que son residuos
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if len(stripped) < 15:
            continue
        if stripped[0] in (",", ";", ")"):
            continue
        lines.append(stripped)
    result = "\n".join(lines).strip()
    # Segundo recorte: si despues de limpiar sigue empezando roto
    result = _trim_broken_start(result)
    return result


def _extract_intro_sentences(text, max_sentences=4):
    """Extrae las primeras oraciones utiles como resumen introductorio."""
    # Separar en oraciones (punto + espacio + mayuscula)
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\xc1\xc9\xcd\xd3\xda\xd1])", text)
    intro = []
    skip_patterns = re.compile(
        r"^(En este art[ií]culo|Tiempo de implementaci|Dificultad:|"
        r"Objetivo:|Cu[aá]ndo utilizarlo|Sigue aprendiendo|"
        r"Te han quedado dudas|Recuerda que tienes|"
        r"Esta estrategia forma parte|Para implementarla)", re.I
    )
    # Parar si encontramos un paso numerado o cabecera de seccion
    step_start = re.compile(r"^(\d+\.\s|PASO\s|Paso\s|Step\s|Instalaci[oó]n\s|Configuraci[oó]n\s)", re.I)
    for s in sentences:
        s = s.strip()
        if not s or len(s) < 25:
            continue
        if skip_patterns.match(s):
            continue
        # Si encontramos inicio de pasos, parar la intro
        if step_start.match(s):
            break
        # Filtrar fragmentos que terminan cortados (sin punto final)
        if len(s) > 50 and not s.endswith((".", "!", "?", ":", ")")):
            # Intentar cortar en el ultimo punto
            last_dot = s.rfind(".")
            if last_dot > 30:
                s = s[:last_dot + 1]
            else:
                continue
        intro.append(s)
        if len(intro) >= max_sentences:
            break
    return " ".join(intro) if intro else ""


def _extract_action_steps(lines):
    """Extrae pasos como acciones concretas, no titulos de seccion."""
    _SECTION_HEADER = re.compile(
        r"^(Nodo\s|Configuraci[oó]n\s|Instalaci[oó]n\s|Funcionamiento|"
        r"Primer\s|Segundo\s|Tercer\s|Rama\s|PASO\s+\d)", re.I
    )
    steps = []
    for line in lines:
        m = _REAL_STEP.match(line)
        m2 = _NAMED_STEP.match(line)
        if m:
            step_text = m.group(2).strip()
        elif m2:
            step_text = m2.group(2).strip()
        else:
            continue
        # Filtrar lineas cortas (titulos, no acciones)
        if len(step_text) < 20:
            continue
        # Filtrar cabeceras de seccion
        if _SECTION_HEADER.match(step_text):
            continue
        # Filtrar si termina cortado (sin punto ni signo de cierre)
        if not step_text.endswith((".", "!", "?", ":", ")", "\"")):
            last_dot = step_text.rfind(".")
            if last_dot > 20:
                step_text = step_text[:last_dot + 1]
            elif len(step_text) < 30:
                continue
        steps.append(step_text)
    return steps


def build_response_md(query, results):
    """Genera respuesta sintetizada: no pega chunks, sino que extrae y estructura."""
    main_result = results[0]
    related = results[1:]

    # Limpiar texto del articulo principal
    main_text = _clean_chunk_for_display(main_result["text"])
    main_lines = [l for l in main_text.split("\n") if l.strip()]
    has_steps = _has_real_steps(main_lines)

    parts = []

    # --- 1) Titulo en cursiva ---
    parts.append(f"*{query.strip()}*\n")

    # --- 2) Respuesta: sintesis de las primeras oraciones ---
    parts.append("**Respuesta:**\n")
    intro = _extract_intro_sentences(main_text)
    if intro:
        parts.append(intro)
    else:
        parts.append(
            f"Connectif cuenta con una guia sobre este tema en el articulo "
            f"*{main_result['title']}*."
        )
    parts.append("")

    # --- 3) Como se hace (SOLO si hay pasos reales) ---
    if has_steps:
        steps = _extract_action_steps(main_lines)
        if steps:
            parts.append("**Como se hace:**\n")
            for i, step in enumerate(steps[:10], 1):
                parts.append(f"{i}. {step}")
            parts.append("")

    # --- 4) Fuentes (1-5 links, sin duplicar) ---
    parts.append("**Fuentes:**\n")
    seen_urls = set()
    all_results = [main_result] + related[:4]
    for r in all_results:
        if r["url"] not in seen_urls:
            seen_urls.add(r["url"])
            parts.append(f"- [{r['title']}]({r['url']})")
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
    """Muestra mesa de ayuda solo si hay al menos un canal configurado."""
    has_email = bool(HELP_EMAIL)
    has_whatsapp = bool(HELP_WHATSAPP)
    if not has_email and not has_whatsapp:
        return
    lines = []
    if has_email:
        lines.append(
            f"<p>Si necesitas ayuda, escribe a: "
            f"<strong>{_html_escape(HELP_EMAIL)}</strong></p>"
        )
    if has_whatsapp:
        lines.append(
            f"<p>WhatsApp: <strong>{_html_escape(HELP_WHATSAPP)}</strong></p>"
        )
    lines.append(
        f'<p><a href="{HELP_FORM}" target="_blank">'
        f"Enviar solicitud al soporte de Connectif</a></p>"
    )
    body = "\n".join(lines)
    st.markdown(
        f'<div class="help-desk-card">'
        f"<h4>Mesa de ayuda</h4>"
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

def _run_search_and_render(query, vectorizer, tfidf_matrix, chunks, diag):
    """Ejecuta la busqueda y renderiza la respuesta UNA sola vez."""
    render_user_bubble(query)
    results = search(query, vectorizer, tfidf_matrix, chunks)

    if not results:
        render_no_results(query)
        render_help_desk()
        return

    # --- Respuesta unica (incluye Fuentes al final) ---
    response_md = build_response_md(query, results)
    render_bot_bubble(response_md)

    # --- Mesa de ayuda (solo si evidencia debil o error, y hay canales) ---
    best_score = results[0]["score"]
    intent = detect_intent(query)
    if best_score < 0.25 or intent == "error":
        render_help_desk()

    # --- Copiar ---
    with st.expander("Copiar respuesta"):
        st.code(response_md, language="markdown")

    # --- Diagnostico: fragments y scores solo aqui ---
    if diag:
        st.markdown(
            '<p class="diag-label">'
            "<strong>Modo diagnostico — Fragments / Scores:</strong></p>",
            unsafe_allow_html=True,
        )
        for r in results:
            st.markdown(f"- `{r['score']:.4f}` — **{r['title']}**")
            with st.expander(f"Fragment: {r['title'][:50]}"):
                st.text(r["text"])


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
        "<p>Asistente de consulta con IA, basado en la documentacion oficial de Connectif.</p>"
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

    # Toggle diagnostico
    diag = st.toggle("Modo diagnostico", value=False)

    # Formulario de pregunta manual
    with st.form("ask", clear_on_submit=False):
        query = st.text_input(
            "Escribe tu pregunta:",
            placeholder="Ej: Como crear un workflow de carrito abandonado?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Preguntar", use_container_width=True)

    # Preguntas frecuentes
    st.markdown(
        '<p class="faq-title">Preguntas frecuentes</p>',
        unsafe_allow_html=True,
    )
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

    # Determinar query activa: FAQ click > form submit
    active_query = ""
    if faq_clicked:
        active_query = faq_clicked
    elif submitted and query.strip():
        active_query = query.strip()

    if not active_query:
        return

    # Ejecutar busqueda y mostrar respuesta
    _run_search_and_render(active_query, vectorizer, tfidf_matrix, chunks, diag)


if __name__ == "__main__":
    main()
