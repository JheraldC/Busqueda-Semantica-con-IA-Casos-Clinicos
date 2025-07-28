# Búsqueda Semántica con IA - Casos Clínicos

Este proyecto implementa un sistema de orientación clínica semántica que permite al usuario ingresar síntomas y características clínicas, y obtener como resultado una lista de casos clínicos similares, un diagnóstico estimado, una explicación médica, la especialidad sugerida y preguntas recomendadas.

Está diseñado en dos módulos principales:

- **Backend:** desarrollado en FastAPI, hace uso del modelo MedGemma 4B para generar orientación médica.
- **Frontend:** desarrollado en React (Vite), permite la interacción del usuario a través de una interfaz intuitiva.

---

## 🔍 Objetivo

Apoyar el diagnóstico clínico preliminar y la formación médica mediante el uso de procesamiento de lenguaje natural (PLN) y recuperación semántica a partir de casos clínicos vectorizados.

---

## 📦 Requisitos del Sistema

### Requisitos Generales (si se ejecuta MedGemma localmente)

- **SO:** Ubuntu 22.04 LTS / Windows 10 o superior (64 bits)
- **CPU:** 6 núcleos (2.6 GHz o más)
- **RAM:** 25 GB
- **Disco:** 50 GB libres
- **Acceso a Internet:** necesario para instalación y autenticación con Hugging Face

### Backend

- Python 3.9 o superior
- pip
- huggingface_hub
- fastapi, uvicorn
- sentence-transformers, pandas, transformers

### Frontend

- Node.js 18+
- npm 9+