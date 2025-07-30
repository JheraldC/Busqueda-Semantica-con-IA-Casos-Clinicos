import json
from pathlib import Path
from fastapi.responses import JSONResponse
from fastapi import FastAPI, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from io import BytesIO
import torch
import re
import httpx

# ========== Carga modelo de embeddings ==========
model_st = SentenceTransformer('all-MiniLM-L6-v2')
df = pd.read_csv("dataset_casos_clinicos_OK.csv", encoding="utf-8")
HISTORIAL_FILE = Path("historial_resultados.json")
embeddings = model_st.encode(df['motivo_consulta'].astype(str) + ". " + df['historia_actual'].astype(str))

# ========== Configuración FastAPI ==========
app = FastAPI()
origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== Modelos de datos ==========
class ConsultaSimple(BaseModel):
    edad: int
    sexo: str
    peso: float = None
    sintomas: str
    antecedentes: str = None

class ConsultaTexto(BaseModel):
    texto: str

# ========== Modelo local de texto ==========
modelo_local = None
tokenizer_local = None
modelo_local_cargado = False

def cargar_modelo_local():
    global modelo_local, tokenizer_local, modelo_local_cargado
    if not modelo_local_cargado:
        try:
            print("Cargando modelo MedGemma localmente...")
            from transformers import AutoTokenizer, AutoModelForCausalLM
            tokenizer_local = AutoTokenizer.from_pretrained("google/medgemma-4b-it")
            modelo_local = AutoModelForCausalLM.from_pretrained("google/medgemma-4b-it")
            modelo_local_cargado = True
        except Exception as e:
            print("No se pudo cargar el modelo local:", str(e))

# ========== Procesamiento de respuesta markdown ==========
def parsear_respuesta_medgemma_markdown(respuesta):
    diagnostico = re.search(r"1\.\s+\*\*Diagn[oó]stico probable:\*\*\s*(.*)", respuesta, re.IGNORECASE)
    explicacion = re.search(r"2\.\s+\*\*Explicaci[oó]n:\*\*\s*(.*)", respuesta, re.IGNORECASE)
    especialidad = re.search(r"3\.\s+\*\*Especialidad m[eé]dica recomendada:\*\*\s*(.*)", respuesta, re.IGNORECASE)
    preguntas = re.search(r"4\.\s+\*\*Preguntas que deber[ií]a hacer el m[eé]dico en consulta:\*\*\s*([\s\S]*)", respuesta, re.IGNORECASE)

    preguntas_texto = ""
    if preguntas and preguntas.group(1):
        preguntas_list = re.findall(r"[*-]\s*(.*)", preguntas.group(1))
        if preguntas_list:
            preguntas_texto = " ".join([q.strip() for q in preguntas_list])
        else:
            preguntas_texto = preguntas.group(1).strip()

    return {
        "diagnostico_probable": diagnostico.group(1).strip() if diagnostico else "",
        "explicacion": explicacion.group(1).strip() if explicacion else "",
        "especialidad_recomendada": especialidad.group(1).strip() if especialidad else "",
        "preguntas_consulta": preguntas_texto
    }

# ========== Consulta a LLM ==========
def consulta_llm(prompt, max_new_tokens=256, usar_nube=True):
    if usar_nube:
        try:
            response = httpx.post(
                "http://35.208.119.20:8000/medgemma",
                json={"texto": prompt},
                timeout=120
            )
            if response.status_code == 200:
                return response.json()["respuesta"]
            else:
                raise Exception("Respuesta inválida desde la nube")
        except Exception as e:
            print("Fallo en nube, usando modelo local:", str(e))

    if not modelo_local_cargado:
        cargar_modelo_local()

    inputs = tokenizer_local(prompt, return_tensors="pt")
    outputs = modelo_local.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer_local.decode(outputs[0], skip_special_tokens=True)

# ========== Endpoints ==========
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
def diagnostico_inteligente(consulta: ConsultaTexto, usar_nube: bool = Query(True)):
    try:
        consulta_emb = model_st.encode([consulta.texto])[0]
        sims = cosine_similarity([consulta_emb], embeddings)[0]
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

        respuesta = consulta_llm(prompt, usar_nube=usar_nube)
        print("DEBUG RESPUESTA:", respuesta)
        return parsear_respuesta_medgemma_markdown(respuesta)
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

# ========== NUEVO: Diagnóstico basado en imagen ==========
@app.post("/diagnostico_imagen")
async def diagnostico_por_imagen(files: List[UploadFile] = File(...)):
    try:
        if len(files) > 3:
            return JSONResponse(status_code=400, content={"error": "Máximo 3 imágenes permitidas"})

        from transformers import pipeline
        pipe = pipeline(
            "image-text-to-text",
            model="google/medgemma-4b-it",
            torch_dtype=torch.bfloat16,
            device="cuda",
        )

        responses = []

        for file in files:
            image = Image.open(BytesIO(await file.read()))

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are an expert radiologist."}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this CT scan"},
                        {"type": "image", "image": image}
                    ]
                }
            ]

            output = pipe(text=messages, max_new_tokens=300)
            responses.append(output[0]["generated_text"][-1]["content"])

        return {"respuestas": responses}

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": str(e)}