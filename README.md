# Guía Inteligente de Connectif

Asistente de consulta con IA basado en la documentación oficial de Connectif, utilizando RAG (Retrieval-Augmented Generation) con TF-IDF.

## 🚀 Características

- **Búsqueda inteligente**: Sistema RAG con TF-IDF para encontrar información relevante en la documentación
- **Respuestas con IA**: Generación de respuestas claras y estructuradas usando GPT-4o-mini
- **UI Premium**: Interfaz elegante y moderna con diseño tipo "guía premium"
- **FAQ interactivo**: Preguntas frecuentes en formato chips que ejecutan búsquedas automáticas
- **Modo diagnóstico**: Opción para ver información técnica sobre las búsquedas

## 📋 Requisitos

- Python 3.8+
- OpenAI API Key (opcional, para respuestas con GPT)

## 🛠️ Instalación

1. Clona el repositorio:
```bash
git clone <tu-repositorio>
cd connectif-bot
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

3. Configura las variables de entorno:
```bash
export OPENAI_API_KEY="tu-api-key"
export HELP_EMAIL="tu-email@ejemplo.com"  # Opcional
```

4. Ingestiona y construye el índice:
```bash
python ingest.py
python build_index.py
```

## 🚀 Ejecución Local

```bash
streamlit run app.py
```

## 📦 Deploy en Streamlit Cloud

1. Sube tu código a GitHub
2. Ve a [Streamlit Cloud](https://streamlit.io/cloud)
3. Conecta tu repositorio
4. Configura las variables de entorno en la sección de Settings:
   - `OPENAI_API_KEY`: Tu clave de API de OpenAI
   - `HELP_EMAIL`: (Opcional) Tu email de contacto
5. Asegúrate de que el archivo principal sea `app.py`
6. El deploy se realizará automáticamente

## 📁 Estructura del Proyecto

```
connectif-bot/
├── app.py              # Aplicación principal Streamlit
├── ingest.py           # Script para ingerir documentos
├── build_index.py      # Script para construir el índice TF-IDF
├── chat.py             # Módulo de chat (si aplica)
├── requirements.txt    # Dependencias Python
├── data/
│   ├── raw/            # Documentos originales
│   └── index/          # Índices generados (chunks, TF-IDF)
└── .streamlit/
    └── config.toml     # Configuración de Streamlit
```

## 🔧 Configuración

### Variables de Entorno

- `OPENAI_API_KEY`: Clave de API de OpenAI (requerida para respuestas con GPT)
- `HELP_EMAIL`: Email de contacto (opcional)

### Logo

Para agregar un logo, coloca una imagen en:
```
connectif-bot/assets/logo.png
```

El sistema detectará automáticamente si existe y lo mostrará en el header.

## 📝 Notas

- El motor de búsqueda RAG utiliza TF-IDF y no requiere modificaciones
- Los índices se generan localmente y deben estar presentes en `data/index/`
- El sistema funciona sin OpenAI API Key, pero mostrará respuestas más básicas

## 📞 Soporte

Para preguntas adicionales, contacta a nuestra [mesa de ayuda](https://support.connectif.ai/hc/es/requests/new).
