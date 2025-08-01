import httpx

url = "http://35.215.215.214:8000/medgemma-img"

prompt = (
    "Caso radiológico. Analiza en español esta imagen médica y proporciona un informe clínico detallado en el siguiente formato (una sola vez):\n"
    "Hallazgos principales: [Describe lesiones, estructuras alteradas, órganos afectados, patrones anormales, signos de inflamación, tumores, fracturas, etc.]\n"
    "Diagnóstico probable: [Hipótesis diagnóstica basada en los hallazgos, no concluyente]\n"
    "Discusión del caso: [Explica el razonamiento, limita los diagnósticos diferenciales, fundamenta tu interpretación]\n"
    "Recomendaciones clínicas: [Sugerencias para manejo, seguimiento o exámenes adicionales]"
)

with open("imagen_1.jpeg", "rb") as f:
    files = {
        "files": ("imagen_1.jpeg", f, "image/jpeg"),
        "prompt": (None, prompt)
    }
    response = httpx.post(url, files=files, timeout=300)  # 5 minutos
    print(response.status_code)
    print(response.json())
