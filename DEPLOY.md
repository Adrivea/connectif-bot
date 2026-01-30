# 🚀 Guía de Deploy - Streamlit Cloud

## ✅ Cambios Realizados

1. ✅ **UI Premium embellecida** - Diseño elegante tipo "guía premium"
2. ✅ **Link actualizado** - `https://support.connectif.ai/hc/es/requests/new` en el footer
3. ✅ **Código subido a GitHub** - Repositorio: `https://github.com/Adrivea/connectif-bot.git`

## 📦 Deploy en Streamlit Cloud

### Paso 1: Acceder a Streamlit Cloud
1. Ve a [https://share.streamlit.io/](https://share.streamlit.io/)
2. Inicia sesión con tu cuenta de GitHub

### Paso 2: Conectar Repositorio
1. Haz clic en **"New app"**
2. Selecciona el repositorio: `Adrivea/connectif-bot`
3. Branch: `master`
4. Main file path: `app.py`

### Paso 3: Configurar Variables de Entorno
En la sección **"Advanced settings"** o **"Secrets"**, agrega:

```
OPENAI_API_KEY=tu-api-key-aqui
HELP_EMAIL=tu-email@ejemplo.com
```

**Nota:** `HELP_EMAIL` es opcional.

### Paso 4: Deploy
1. Haz clic en **"Deploy!"**
2. Streamlit Cloud construirá y desplegará tu app automáticamente
3. La URL será: `https://tu-app.streamlit.app`

## 🔧 Requisitos Previos

Asegúrate de que los archivos de índice estén en el repositorio:
- `data/index/chunks.joblib`
- `data/index/tfidf_matrix.joblib`
- `data/index/vectorizer.joblib`

**Importante:** Si los índices son muy grandes, considera:
1. Subirlos a un servicio de almacenamiento (S3, Google Cloud Storage)
2. Generarlos durante el deploy usando un script de setup
3. Usar Git LFS para archivos grandes

## 📝 Verificación Post-Deploy

1. ✅ Verifica que la UI se vea correctamente
2. ✅ Prueba una búsqueda con el FAQ
3. ✅ Verifica que el link de "mesa de ayuda" funcione
4. ✅ Prueba el modo diagnóstico
5. ✅ Verifica que las respuestas se generen correctamente

## 🐛 Troubleshooting

### Error: "No se encontro el indice"
- Asegúrate de que los archivos `.joblib` estén en `data/index/`
- Verifica que estén incluidos en el repositorio (no en `.gitignore`)

### Error: "OPENAI_API_KEY not found"
- Verifica que la variable de entorno esté configurada en Streamlit Cloud
- Revisa que el nombre sea exactamente `OPENAI_API_KEY`

### La app no carga
- Revisa los logs en Streamlit Cloud
- Verifica que `requirements.txt` tenga todas las dependencias
- Asegúrate de que `app.py` esté en la raíz del repositorio

## 🔗 Links Importantes

- **Repositorio GitHub:** https://github.com/Adrivea/connectif-bot
- **Mesa de Ayuda:** https://support.connectif.ai/hc/es/requests/new
- **Streamlit Cloud:** https://share.streamlit.io/
