"""
app.py — Asistente Connectif (Docs) — Interfaz web con Streamlit.

Reutiliza el indice TF-IDF de data/index/ y presenta respuestas
en lenguaje natural con tono de agente de soporte.
"""

import os
import re
import textwrap

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
    """Limpia un fragmento: quita saltos excesivos y espacios multiples."""
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _deduplicate_sentences(fragments):
    """Fusiona fragmentos eliminando oraciones repetidas o muy similares."""
    seen = set()
    unique_lines = []
    for frag in fragments:
        clean = _clean_fragment(frag)
        for line in clean.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Normalizar para comparar: minusculas, sin puntuacion extra
            key = re.sub(r"[^\w\s]", "", line.lower()).strip()
            if len(key) < 15:
                continue  # lineas muy cortas no aportan
            if key not in seen:
                seen.add(key)
                unique_lines.append(line)
    return unique_lines


def _summarize_lines(lines, max_lines=12):
    """Toma las lineas mas relevantes, sin exceder el maximo."""
    return lines[:max_lines]

# ---------------------------------------------------------------------------
# Construccion de respuesta con formato obligatorio
# ---------------------------------------------------------------------------

def build_response(query, results):
    """Genera la respuesta completa en markdown siguiendo el formato obligatorio."""
    intent = detect_intent(query)

    # Fusionar y deduplicar fragmentos
    all_fragments = [r["text"] for r in results]
    unique_lines = _deduplicate_sentences(all_fragments)
    summarized = _summarize_lines(unique_lines)

    # Titulos de articulos encontrados (para contexto)
    titles = list(dict.fromkeys(r["title"] for r in results))

    parts = []

    # --- Titulo ---
    parts.append(f"### {query.strip().rstrip('?')}?\n")

    # --- Explicacion general ---
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
            f"Encontre informacion relevante sobre tu consulta "
            f"en la documentacion de Connectif."
        )
        if len(titles) > 0:
            parts.append(
                f"Los articulos mas relacionados son: **{titles[0]}**"
                + (f" y *{titles[1]}*." if len(titles) > 1 else ".")
            )
        parts.append("")

    # --- Pasos / Contenido principal ---
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

    # --- Que debes tener en cuenta ---
    if len(summarized) > 8:
        parts.append("#### Que debes tener en cuenta:\n")
        for line in summarized[8:12]:
            parts.append(f"- {line}")
        parts.append("")

    # --- Checklist rapido (solo para "how") ---
    if intent == "how" and len(titles) > 0:
        parts.append("#### Checklist rapido:\n")
        checks = [f"- [ ] Revisa el articulo: {t}" for t in titles[:3]]
        if intent == "how":
            checks.append("- [ ] Verifica que los cambios se guardaron correctamente")
            checks.append("- [ ] Prueba el resultado en tu cuenta de Connectif")
        for c in checks:
            parts.append(c)
        parts.append("")

    # --- Fuentes (obligatorio) ---
    parts.append("---\n")
    parts.append("### Fuentes\n")
    seen_urls = set()
    for r in results[:5]:
        if r["url"] not in seen_urls:
            seen_urls.add(r["url"])
            parts.append(f"- [{r['title']}]({r['url']})")
    parts.append("")

    # --- Mesa de ayuda (si la evidencia es debil) ---
    best_score = results[0]["score"] if results else 0
    if best_score < 0.25 or intent == "error":
        parts.append(_mesa_de_ayuda_block())

    return "\n".join(parts)


def _mesa_de_ayuda_block():
    """Genera el bloque de mesa de ayuda en markdown."""
    lines = [
        "---\n",
        "### Mesa de ayuda\n",
        "Si necesitas mas ayuda, puedes contactar al equipo de soporte de Connectif:\n",
    ]
    if HELP_EMAIL:
        lines.append(f"- Email: **{HELP_EMAIL}**")
    if HELP_WHATSAPP:
        lines.append(f"- WhatsApp: **{HELP_WHATSAPP}**")
    lines.append(f"- [Enviar solicitud al soporte de Connectif]({HELP_FORM})")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Respuesta cuando no hay resultados
# ---------------------------------------------------------------------------

def build_no_results_response(query):
    parts = []
    parts.append(f"### {query.strip().rstrip('?')}?\n")
    parts.append(
        "**No encontre esta informacion en la documentacion oficial de Connectif.**\n"
    )
    parts.append("Intenta reformular tu pregunta:\n")
    parts.append("- Usa palabras clave especificas (ej: *workflow*, *segmento*, *email*)")
    parts.append("- Describe la accion que quieres realizar (ej: *como crear...*, *como configurar...*)")
    parts.append("- Menciona la seccion de Connectif (ej: *editor de email*, *cupones*, *integracion*)")
    parts.append("")
    parts.append(_mesa_de_ayuda_block())
    return "\n".join(parts)

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Asistente Connectif (Docs)",
        page_icon=":blue_book:",
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
        # Respuesta principal formateada
        response_md = build_response(query, results)
        st.markdown(response_md)

        # Boton copiar respuesta
        with st.expander("Copiar respuesta"):
            st.code(response_md, language="markdown")

        # Diagnostico
        if diag:
            st.markdown("---")
            st.markdown("**Modo diagnostico — Fragments / Scores:**")
            for r in results:
                st.markdown(f"- `{r['score']:.4f}` — **{r['title']}**")
                with st.expander(f"Fragment: {r['title'][:50]}"):
                    st.text(r["text"])
    else:
        response_md = build_no_results_response(query)
        st.markdown(response_md)


if __name__ == "__main__":
    main()
