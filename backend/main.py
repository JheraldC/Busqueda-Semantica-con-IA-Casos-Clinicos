import json
from pathlib import Path
from fastapi.responses import JSONResponse
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_csv("dataset_casos_clinicos_OK.csv", encoding="utf-8")
HISTORIAL_FILE = Path("historial_resultados.json")

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings = model.encode(df['motivo_consulta'].astype(str) + ". " + df['historia_actual'].astype(str))

# Modelo para consulta estructurada
class ConsultaSimple(BaseModel):
    edad: int
    sexo: str
    peso: float = None
    sintomas: str
    antecedentes: str = None

# Modelo para texto libre
class ConsultaTexto(BaseModel):
    texto: str

def consulta_ollama(prompt, modelo="deepseek-r1:8b"):
    import requests
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": modelo,
            "prompt": prompt,
            "stream": False
        }
    )
    data = response.json()
    texto_completo = data.get("response", "")

    # Extraer solo la parte posterior a </think>
    if "</think>" in texto_completo:
        texto_limpio = texto_completo.split("</think>")[-1].strip()
    else:
        texto_limpio = texto_completo.strip()

    return texto_limpio

# POST Buscar de estructura
@app.post("/buscar")
def buscar_casos(consulta: ConsultaSimple):
    try:
        texto = (
            f"Edad: {consulta.edad}. "
            f"Sexo: {consulta.sexo}. "
            f"Peso: {consulta.peso or ''}. "
            f"Síntomas: {consulta.sintomas}. "
            f"Antecedentes: {consulta.antecedentes or ''}."
        )

        consulta_emb = model.encode([texto])[0]
        sims = cosine_similarity([consulta_emb], embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:5]
        resultados = df.iloc[top_idx].where(pd.notnull(df.iloc[top_idx]), None).to_dict(orient="records")
        return {"resultados": resultados}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": str(e)}

# POST Buscar de un texto
@app.post("/buscar_texto")
def buscar_casos_texto(consulta: ConsultaTexto):
    try:
        consulta_emb = model.encode([consulta.texto])[0]
        sims = cosine_similarity([consulta_emb], embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:5]
        resultados = df.iloc[top_idx].where(pd.notnull(df.iloc[top_idx]), None).to_dict(orient="records")
        return {"resultados": resultados}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": str(e)}

# POST Buscar con Deepseek r1
import re

@app.post("/diagnostico_inteligente")
def diagnostico_inteligente(consulta: ConsultaTexto):
    try:
        # Busca casos similares
        consulta_emb = model.encode([consulta.texto])[0]
        sims = cosine_similarity([consulta_emb], embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:5]
        casos_similares = df.iloc[top_idx]

        # Construye prompt para Ollama
        prompt = (
            "Eres un asistente médico IA. Analiza este caso clínico:\n\n"
            f"Paciente: {consulta.texto}\n\n"
            "Basándote en estos 5 casos clínicos similares, "
            "haz un diagnóstico probable, una breve explicación clara (máximo 3 frases) "
            "que incluya una pequeña descripción de la enfermedad, "
            "la especialidad médica recomendada y preguntas que el médico debería hacer en la consulta.\n\n"
            "Casos similares:\n"
        )
        for i, row in casos_similares.iterrows():
            prompt += (
                f"- Caso {i+1}: Motivo de consulta: {row['motivo_consulta']}. "
                f"Historia actual: {row['historia_actual']}. "
                f"Diagnóstico: {row['diagnostico']}.\n"
            )

        prompt += (
            "\nPor favor, responde solo con un texto estructurado similar a este formato:\n"
            "Diagnóstico probable: <diagnóstico>.\n"
            "Explicación: <explicación breve>.\n"
            "Especialidad recomendada: <especialidad>.\n"
            "Preguntas para la consulta: <preguntas>.\n"
        )

        respuesta = consulta_ollama(prompt, modelo="deepseek-r1:8b")

        # Parsear la respuesta en campos usando expresiones regulares
        diagnostico_probable = re.search(r"Diagnóstico probable:\s*(.*)", respuesta)
        explicacion = re.search(r"Explicación:\s*(.*)", respuesta)
        especialidad = re.search(r"Especialidad recomendada:\s*(.*)", respuesta)
        preguntas = re.search(r"Preguntas para la consulta:\s*([\s\S]*)", respuesta)  # Captura multilinea

        return {
            "diagnostico_probable": diagnostico_probable.group(1).strip() if diagnostico_probable else "",
            "explicacion": explicacion.group(1).strip() if explicacion else "",
            "especialidad_recomendada": especialidad.group(1).strip() if especialidad else "",
            "preguntas_consulta": preguntas.group(1).strip() if preguntas else ""
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": str(e)}

@app.post("/guardarResultado")
async def guardar_resultado(resultado: dict):
    try:
        # Leer archivo existente o crear lista vacía
        if HISTORIAL_FILE.exists():
            with open(HISTORIAL_FILE, "r", encoding="utf-8") as f:
                historial = json.load(f)
        else:
            historial = []

        # Añadir nuevo resultado
        historial.append(resultado)

        # Guardar actualizado
        with open(HISTORIAL_FILE, "w", encoding="utf-8") as f:
            json.dump(historial, f, indent=2, ensure_ascii=False)

        return {"mensaje": "Resultado guardado exitosamente"}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/resultados")
async def obtener_resultados():
    try:
        if HISTORIAL_FILE.exists():
            with open(HISTORIAL_FILE, "r", encoding="utf-8") as f:
                historial = json.load(f)
        else:
            historial = []
        return {"resultados": historial}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})