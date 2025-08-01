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
from typing import List
import httpx
import pytz
from datetime import datetime
from transformers import pipeline
import time

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

class SimilitudOracionesRequest(BaseModel):
    source_sentence: str
    sentences: list[str]

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

# ========== Procesamiento de limpiar respuestas ==========
def limpiar_bloque(texto):
    if not texto:
        return ""
    texto = re.sub(r'^[*\-\s]+', '', texto, flags=re.MULTILINE)
    texto = re.sub(r':\*\*', '', texto)
    texto = texto.replace('\n', ' ').strip()
    return texto

# ========== Procesamiento de respuesta individual imagen ==========
def parsear_informe_radiologico(descripcion):
    # Variantes: con o sin asteriscos, minúsculas/mayúsculas, etc.
    patt_hallazgos = r'(\*\*)?\s*hallazgos principales\s*:?(\*\*)?\s*([\s\S]*?)(\*\*diagn[oó]stico probable\s*:?\*\*|diagn[oó]stico probable\s*:|$)'
    patt_diagnostico = r'(\*\*)?\s*diagn[oó]stico probable\s*:?(\*\*)?\s*([\s\S]*?)(\*\*discusi[oó]n|discusi[oó]n|$)'

    hallazgos = re.search(patt_hallazgos, descripcion, re.IGNORECASE)
    diagnostico = re.search(patt_diagnostico, descripcion, re.IGNORECASE)

    hallazgos_text = limpiar_bloque(hallazgos.group(3)) if hallazgos else ""
    diagnostico_text = limpiar_bloque(diagnostico.group(3)) if diagnostico else ""

    if not hallazgos_text and not diagnostico_text:
        return {
            "hallazgos_principales": "",
            "diagnostico_probable": "",
            "texto_bruto": descripcion
        }

    return {
        "hallazgos_principales": hallazgos_text,
        "diagnostico_probable": diagnostico_text,
    }

# ========== Procesamiento de respuesta final imagen ==========
import re

def parsear_diagnostico_global(texto):
    """
    Extrae el ÚLTIMO bloque 'Hallazgo final' y 'Diagnóstico global' del texto generado por el modelo.
    Si no encuentra, devuelve los textos entre corchetes.
    """
    patron = r"Hallazgo final: ?(.+?)\s*Diagnóstico global: ?(.+?)(?:\n|$)"
    matches = list(re.finditer(patron, texto, re.DOTALL))
    if matches:
        # Tomar el último bloque (el real, después del ejemplo)
        ultimo = matches[-1]
        hallazgo = ultimo.group(1).strip()
        diagnostico = ultimo.group(2).strip()
    else:
        hallazgo = "[Resumen de los hallazgos de todas las imágenes]"
        diagnostico = "[Diagnóstico integrador, justifica brevemente la hipótesis y posibles diagnósticos diferenciales]"
    return {
        "hallazgo_final": hallazgo,
        "diagnostico_global": diagnostico
    }
    
# ========== Procesamiento de respuesta markdown ==========
def parsear_respuesta_medgemma_markdown(respuesta):
    # Busca TODAS las ocurrencias del bloque de diagnóstico
    bloques = list(re.finditer(
        r"1\.\s+\*\*Diagn[oó]stico probable:\*\*\s*(.*?)\s*"
        r"2\.\s+\*\*Explicaci[oó]n:\*\*\s*(.*?)\s*"
        r"3\.\s+\*\*Especialidad m[eé]dica recomendada:\*\*\s*(.*?)\s*"
        r"4\.\s+\*\*Preguntas que deber[ií]a hacer el m[eé]dico en consulta:\*\*\s*([\s\S]*?)(?=\n\d\.|\Z)",
        respuesta, re.IGNORECASE | re.DOTALL
    ))

    if not bloques:
        return {
            "diagnostico_probable": "",
            "explicacion": "",
            "especialidad_recomendada": "",
            "preguntas_consulta": ""
        }

    # Selecciona el SEGUNDO bloque si existe, sino el primero
    match = bloques[1] if len(bloques) > 1 else bloques[0]
    diagnostico = match.group(1).strip()
    explicacion = match.group(2).strip()
    especialidad = match.group(3).strip()
    preguntas_raw = match.group(4).strip()

    preguntas_list = re.findall(r"[*-]\s*(.*)", preguntas_raw)
    preguntas_list = [q.strip() for q in preguntas_list if q.strip()][:5]
    if len(preguntas_list) < 3 and preguntas_raw:
        # Intenta dividir por salto de línea o punto y coma si no hay guiones
        extra = re.split(r'[\n;]+', preguntas_raw)
        preguntas_list += [q.strip() for q in extra if q.strip()]
        preguntas_list = preguntas_list[:5]
    preguntas_texto = " | ".join(preguntas_list)

    return {
        "diagnostico_probable": diagnostico,
        "explicacion": explicacion,
        "especialidad_recomendada": especialidad,
        "preguntas_consulta": preguntas_texto
    }

# ========== Verifica si estamos en horario de nube ==========
def en_horario_nube():
    hora_actual = datetime.now().hour
    return 7 <= hora_actual < 18

# ========== Consulta a LLM usando nube con fallback ==========
def consulta_llm(prompt, max_new_tokens=256, usar_nube=True):
    respuesta_final = ""
    advertencia = ""
    debug_msg = "[DEBUG-LLM]"

    print(f"{debug_msg} Nuevo request:")
    print(f"{debug_msg} usar_nube={usar_nube}, en_horario_nube={en_horario_nube()}")
    print(f"{debug_msg} Prompt enviado (primeros 120 chars): {prompt[:120]}...")

    if usar_nube and en_horario_nube():
        inicio = time.time()
        try:
            response = httpx.post(
                "http://35.215.215.214:8000/medgemma",
                json={"texto": prompt},
                timeout=None
            )
            t_total = time.time() - inicio
            print(f"{debug_msg} [NUBE] status_code={response.status_code} tiempo={t_total:.2f}s")

            if response.status_code == 200:
                print(f"{debug_msg} [NUBE] Respuesta OK, devolviendo resultado.")
                return response.json()["respuesta"]
            else:
                print(f"{debug_msg} [NUBE] Error HTTP status {response.status_code}. Activando fallback local.")
                raise Exception(f"Respuesta inválida desde la nube. Status: {response.status_code}")
        except Exception as e:
            t_total = time.time() - inicio
            print(f"{debug_msg} [NUBE] EXCEPCIÓN o TIMEOUT tras {t_total:.2f}s: {e}")
            advertencia = f"⚠️ No se pudo usar la nube: {e}. Usando modelo local como respaldo."
    elif usar_nube and not en_horario_nube():
        print(f"{debug_msg} [LOCAL] Fuera de horario de nube, usando local.")
        advertencia = "ℹ️ Estás fuera del horario permitido para el uso de la nube (7:00 a.m. - 11:00 p.m.). Usando modelo local."

    print(f"{debug_msg} [LOCAL] Usando modelo local realmente.")
    if not modelo_local_cargado:
        cargar_modelo_local()
    inputs = tokenizer_local(prompt, return_tensors="pt")
    outputs = modelo_local.generate(**inputs, max_new_tokens=max_new_tokens)
    respuesta_final = tokenizer_local.decode(outputs[0], skip_special_tokens=True)

    if advertencia:
        respuesta_final = f"{advertencia}\n\n{respuesta_final}"

    print(f"{debug_msg} [LOCAL] Respuesta local generada OK.")
    return respuesta_final

# ========== Endpoints ==========

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
    
# ========== Diagnóstico basado en imagen ==========
@app.post("/diagnostico_imagen")
async def diagnostico_por_imagen(files: List[UploadFile] = File(...)):
    import time
    inicio = time.time()
    usar_nube = en_horario_nube()  # O puedes forzar True/False para pruebas
    try:
        if len(files) > 3:
            return JSONResponse(status_code=400, content={"error": "Máximo 3 imágenes permitidas"})

        results = []
        informes_solo_texto = []

        prompt_individual = (
            "Analiza en español esta imagen médica y responde estrictamente en el siguiente formato, sin agregar texto extra:\n"
            "Hallazgos principales: [Describe solo los hallazgos relevantes, como lesiones, anomalías, órganos afectados.]\n"
            "Diagnóstico probable: [Hipótesis diagnóstica basada en los hallazgos, bien detallado. Sé tan detallado como lo permitan los hallazgos en la imagen, usando terminología médica precisa, pero no inventes información no visible.]"
        )

        for file in files:
            image_bytes = await file.read()
            procesado_nube = False
            descripcion = ""
            informe_parseado = {}

            # ========== Intento en la nube ==========
            if usar_nube:
                try:
                    response = httpx.post(
                        "http://35.215.215.214:8000/medgemma-img",
                        files={'files': (file.filename, image_bytes, file.content_type)},
                        data={'prompt': prompt_individual},
                        timeout=300
                    )
                    if response.status_code == 200:
                        data = response.json()
                        if "resultados" in data and len(data["resultados"]) > 0 and "descripcion" in data["resultados"][0]:
                            descripcion = data["resultados"][0]["descripcion"]
                            informe_parseado = parsear_informe_radiologico(descripcion)
                            informes_solo_texto.append(descripcion)
                            procesado_nube = True
                        else:
                            descripcion = "⚠️ Respuesta de la nube con formato inesperado."
                            informe_parseado = {}
                            informes_solo_texto.append(descripcion)
                            procesado_nube = True
                        results.append({
                            "imagen": file.filename,
                            "descripcion": informe_parseado,
                        })
                    else:
                        descripcion = f"Error HTTP {response.status_code}"
                        informe_parseado = {}
                except Exception as e:
                    print("⚠️ Error usando la nube, usando modelo local. Excepción:", str(e))
            
            # ========== Fallback LOCAL ==========
            if not procesado_nube:
                try:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    pipe = pipeline(
                        "image-text-to-text",
                        model="google/medgemma-4b-it",
                        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                        device=device,
                    )
                    messages = [
                        {
                            "role": "system",
                            "content": [{"type": "text", "text": "Eres un radiólogo clínico experto. Responde en español y utiliza terminología médica adecuada."}]
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt_individual},
                                {"type": "image", "image": Image.open(BytesIO(image_bytes))}
                            ]
                        }
                    ]

                    output = pipe(text=messages, max_new_tokens=400)
                    descripcion = output[0]["generated_text"][-1]["content"] if isinstance(output[0]["generated_text"], list) else output[0]["generated_text"]
                    informe_parseado = parsear_informe_radiologico(descripcion)
                    informes_solo_texto.append(descripcion)
                    results.append({
                        "imagen": file.filename,
                        "descripcion": informe_parseado,
                    })
                except Exception as e:
                    results.append({
                        "imagen": file.filename,
                        "descripcion": f"Error procesando imagen en modelo local: {str(e)}",
                    })

        # =========== Diagnóstico global ===========
        diagnostico_global = None
        if len(informes_solo_texto) > 1:
            texto_informes = "\n\n".join([
                f"Informe de la imagen {i+1}:\n{desc}" for i, desc in enumerate(informes_solo_texto)
            ])
            prompt_global = (
                f"Se han analizado varias imágenes médicas, cada una con su informe individual:\n"
                f"{texto_informes}\n\n"
                "Como radiólogo clínico experto, integra todos los hallazgos y redacta un único informe resumen. "
                "Responde exactamente así (NO repitas el formato):\n"
                "Hallazgo final: [Resumen de los hallazgos de todas las imágenes]\n"
                "Diagnóstico global: [Diagnóstico integrador, justifica brevemente la hipótesis y posibles diagnósticos diferenciales]"
            )
            try:
                diagnostico_global = consulta_llm(prompt_global, usar_nube=usar_nube)
                global_parseado = parsear_diagnostico_global(diagnostico_global)
            except Exception as e:
                diagnostico_global = f"Error al generar el diagnóstico global: {str(e)}"
                global_parseado = {"hallazgo_final": "", "diagnostico_global": diagnostico_global}
        else:
            descripcion = results[0]["descripcion"]
            global_parseado = {
                "hallazgo_final": descripcion.get("hallazgos_principales", ""),
                "diagnostico_global": descripcion.get("diagnostico_probable", "")
            }
            diagnostico_global = (
                f"Hallazgo final: {global_parseado['hallazgo_final']}\n\n"
                f"Diagnóstico global: {global_parseado['diagnostico_global']}"
            )

        t_total = time.time() - inicio
        print(f"[TIEMPO] /diagnostico_imagen: {t_total:.2f} segundos")
        print(results)
        print(diagnostico_global)
        return {
            "resultados": results,
            "hallazgo_final": global_parseado["hallazgo_final"],
            "diagnostico_global": global_parseado["diagnostico_global"],
            "diagnostico_global_raw": diagnostico_global
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": str(e)}
    
@app.post("/diagnostico_inteligente")
def diagnostico_inteligente(consulta: ConsultaTexto, usar_nube: bool = Query(True)):
    inicio = time.time()
    try:
        # 1. Embedding y similitud
        consulta_emb = model_st.encode([consulta.texto])[0]
        sims = cosine_similarity([consulta_emb], embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:3]
        casos_similares = df.iloc[top_idx]

        # 2. Construcción del prompt con casos similares
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
        
        # 3. Indicaciones y formato para el modelo (fuera del for)
        prompt += (
            "\nCon base en esta información ten atencion en lo dicho por el paciente en (Edad, Sexo, Peso, Síntomas, Antecedentes) o (Texto de consulta), responde SOLO en español y usando EXACTAMENTE el siguiente formato:\n"
            "1. **Diagnóstico probable:**\n"
            "2. **Explicación:**\n"
            "3. **Especialidad médica recomendada:**\n"
            "4. **Preguntas que debería hacer el médico en consulta:**\n"
            "- Pregunta 1\n"
            "Las preguntas deber ser diferentes, concisas y que sean 5, no más"
            
        )

        # 4. Enviar prompt al modelo (nube o local)
        respuesta = consulta_llm(prompt, usar_nube=usar_nube)
        print("DEBUG RESPUESTA:", respuesta)
        t_total = time.time() - inicio
        print(f"[TIEMPO] /diagnostico_inteligente: {t_total:.2f} segundos")
        return parsear_respuesta_medgemma_markdown(respuesta)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": str(e)}

@app.post("/similitud_oraciones")
def similitud_oraciones(payload: SimilitudOracionesRequest):
    """
    Endpoint para calcular la similitud semántica entre una oración fuente y una lista de oraciones.
    Devuelve scores de similitud en el rango [0, 1].
    """
    try:
        # Puedes cambiar el modelo por uno multilingüe o biomédico si deseas
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        # model = SentenceTransformer('BAAI/bge-m3')  # Ejemplo de modelo multilingüe
        # model = SentenceTransformer('PlanTL-GOB-ES/roberta-base-biomedical-clinical-es')  # Si lo quieres probar biomédico español

        source_emb = model.encode(payload.source_sentence, convert_to_tensor=True)
        targets_emb = model.encode(payload.sentences, convert_to_tensor=True)
        from sentence_transformers import util
        # Scores: mayor a 0.7 ≈ muy parecidos; 0.3-0.7 parecidos; <0.3 poco relacionados
        scores = util.pytorch_cos_sim(source_emb, targets_emb)[0]
        # Convertimos tensor a float por compatibilidad JSON
        print([float(s) for s in scores])
        return {"scores": [float(s) for s in scores]}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": str(e)})