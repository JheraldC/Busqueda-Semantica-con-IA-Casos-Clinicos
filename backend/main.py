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

# ======= Carga MedGemma-4B-Instruct =======
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/medgemma-4b-it")
model = AutoModelForCausalLM.from_pretrained("google/medgemma-4b-it")
# model = model.to("cuda")  # Si tienes GPU compatible

app = FastAPI()
model_st = SentenceTransformer('all-MiniLM-L6-v2')
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

embeddings = model_st.encode(df['motivo_consulta'].astype(str) + ". " + df['historia_actual'].astype(str))

class ConsultaSimple(BaseModel):
    edad: int
    sexo: str
    peso: float = None
    sintomas: str
    antecedentes: str = None

class ConsultaTexto(BaseModel):
    texto: str

def consulta_llm(prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    respuesta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return respuesta

def parsear_respuesta_medgemma_markdown(respuesta):
    # Busca la respuesta a partir de '1.  **Diagnóstico probable:**'
    diagnostico = re.search(r"1\.\s+\*\*Diagn[oó]stico probable:\*\*\s*(.*)", respuesta, re.IGNORECASE)
    explicacion = re.search(r"2\.\s+\*\*Explicaci[oó]n:\*\*\s*(.*)", respuesta, re.IGNORECASE)
    especialidad = re.search(r"3\.\s+\*\*Especialidad m[eé]dica recomendada:\*\*\s*(.*)", respuesta, re.IGNORECASE)
    preguntas = re.search(r"4\.\s+\*\*Preguntas que deber[ií]a hacer el m[eé]dico en consulta:\*\*\s*([\s\S]*)", respuesta, re.IGNORECASE)

    # Procesa preguntas en formato de lista markdown
    preguntas_texto = ""
    if preguntas and preguntas.group(1):
        # Busca todas las líneas que empiezan con *, - o número
        preguntas_list = re.findall(r"[*-]\s*(.*)", preguntas.group(1))
        if preguntas_list:
            preguntas_texto = " ".join([q.strip() for q in preguntas_list])
        else:
            # Si no, toma el bloque tal cual
            preguntas_texto = preguntas.group(1).strip()

    return {
        "diagnostico_probable": diagnostico.group(1).strip() if diagnostico else "",
        "explicacion": explicacion.group(1).strip() if explicacion else "",
        "especialidad_recomendada": especialidad.group(1).strip() if especialidad else "",
        "preguntas_consulta": preguntas_texto
    }


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
        consulta_emb = model_st.encode([texto])[0]
        sims = cosine_similarity([consulta_emb], embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:5]
        resultados = df.iloc[top_idx].where(pd.notnull(df.iloc[top_idx]), None).to_dict(orient="records")
        return {"resultados": resultados}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": str(e)}

@app.post("/buscar_texto")
def buscar_casos_texto(consulta: ConsultaTexto):
    try:
        consulta_emb = model_st.encode([consulta.texto])[0]
        sims = cosine_similarity([consulta_emb], embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:5]
        resultados = df.iloc[top_idx].where(pd.notnull(df.iloc[top_idx]), None).to_dict(orient="records")
        return {"resultados": resultados}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": str(e)}

@app.post("/diagnostico_inteligente")
def diagnostico_inteligente(consulta: ConsultaTexto):
    try:
        consulta_emb = model_st.encode([consulta.texto])[0]
        sims = cosine_similarity([consulta_emb], embeddings)[0]
        # SOLO LOS 3 más parecidos
        top_idx = np.argsort(sims)[::-1][:3]
        casos_similares = df.iloc[top_idx]

        prompt = (
            f"Paciente: {consulta.texto}\n"
            "A continuación, se muestran algunos casos clínicos similares:\n"
        )
        for i, row in casos_similares.iterrows():
            prompt += (
                f"- Caso {i+1}: Motivo de consulta: {row['motivo_consulta']}. "
                f"Historia actual: {row['historia_actual']}. "
                f"Diagnóstico: {row['diagnostico']}.\n"
            )
        prompt += (
            "\nCon base en esta información, responde las siguientes preguntas en español:\n"
            "1. ¿Cuál es el diagnóstico probable?\n"
            "2. Explica brevemente por qué.\n"
            "3. ¿Qué especialidad médica se recomienda?\n"
            "4. ¿Qué preguntas debería hacer el médico en consulta?\n"
        )

        respuesta = consulta_llm(prompt, max_new_tokens=256)
        print("DEBUG RESPUESTA:", respuesta)
        respuesta_parseada = parsear_respuesta_medgemma_markdown(respuesta)
        return respuesta_parseada


    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": str(e)}

@app.post("/guardarResultado")
async def guardar_resultado(resultado: dict):
    try:
        if HISTORIAL_FILE.exists():
            with open(HISTORIAL_FILE, "r", encoding="utf-8") as f:
                historial = json.load(f)
        else:
            historial = []
        historial.append(resultado)
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
