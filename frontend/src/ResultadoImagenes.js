import React, { useState } from "react";
import { useParams } from "react-router-dom";

export default function ResultadoImagen({ resultadosHistorial }) {
    const { id } = useParams();
    const [diagnosticoReal, setDiagnosticoReal] = useState("");
    const [similitudAvanzada, setSimilitudAvanzada] = useState(null);
    const [loadingAvanzada, setLoadingAvanzada] = useState(false);

    // Buscar resultado por ID y asegurarse que sea tipo imagen
    const resultado = resultadosHistorial.find(
        (r) => r.id === id && r.tipoConsulta === "diagnostico_imagen"
    );
    if (!resultado) {
        return <p className="text-red-600">Resultado de imagen no encontrado.</p>;
    }

    // Unir el diagnóstico final de IA
    const diagnosticoIA = [
        resultado.resultado?.hallazgo_final,
        resultado.resultado?.diagnostico_global
    ].filter(Boolean).join("\n\n");

    // Advertencia personalizada según score
    let alerta = null;
    if (similitudAvanzada !== null) {
        if (similitudAvanzada > 0.8) {
            alerta = "¡Diagnósticos prácticamente iguales!";
        } else if (similitudAvanzada > 0.6) {
            alerta = "Diagnósticos con buena coincidencia semántica.";
        } else if (similitudAvanzada < 0.3) {
            alerta = "Diagnósticos completamente diferentes.";
        } else {
            alerta = "Similitud intermedia, revisar cuidadosamente.";
        }
    }

    // Función para similitud avanzada
    async function calcularSimilitudAvanzada() {
        setLoadingAvanzada(true);
        setSimilitudAvanzada(null);
        try {
            const res = await fetch("http://localhost:8000/similitud_oraciones", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    source_sentence: diagnosticoIA,
                    sentences: [diagnosticoReal]
                })
            });
            const data = await res.json();
            let valorSimilitud = null;
            if (data && Array.isArray(data.scores) && data.scores.length > 0) {
                valorSimilitud = Number(data.scores[0]);
            }
            setSimilitudAvanzada(valorSimilitud);

        } catch (err) {
            setSimilitudAvanzada(null);
            alert("Error al consultar similitud avanzada");
        } finally {
            setLoadingAvanzada(false);
        }
    }

    return (
        <section className="max-w-4xl mx-auto bg-white rounded shadow p-8 my-8">
            <h2 className="text-2xl font-bold mb-6 text-indigo-800">Comparación de Diagnóstico IA vs Real</h2>
            <div className="flex gap-8 items-center">
                {/* Diagnóstico IA */}
                <div className="flex-1">
                    <h3 className="font-semibold text-indigo-700 mb-2">Diagnóstico IA</h3>
                    <div className="p-4 bg-indigo-50 rounded shadow-sm whitespace-pre-line mb-2">
                        <strong>Hallazgo final:</strong>
                        <br />
                        {resultado.resultado?.hallazgo_final || "No hay hallazgo final generado por IA."}
                    </div>
                    <div className="p-4 bg-indigo-50 rounded shadow-sm whitespace-pre-line">
                        <strong>Diagnóstico global:</strong>
                        <br />
                        {resultado.resultado?.diagnostico_global || "No hay diagnóstico global generado por IA."}
                    </div>
                </div>

                <div className="flex flex-col items-center justify-center min-w-[210px]">
                    {/* Similitud avanzada */}
                    {similitudAvanzada !== null && !isNaN(similitudAvanzada) && (
                        <div className="mb-4 w-full flex flex-col items-center">
                            <div className="bg-green-50 border border-green-400 rounded-lg px-6 py-3 mb-2 flex flex-col items-center w-full">
                                <span className="text-green-700 text-lg font-semibold">Similitud avanzada:</span>
                                <span className="text-4xl font-extrabold text-green-800 mt-1">
                                    {Math.round(similitudAvanzada * 100)}%
                                </span>
                            </div>
                        </div>
                    )}

                    <button
                        onClick={calcularSimilitudAvanzada}
                        className="mb-2 px-4 py-2 bg-green-700 hover:bg-green-900 text-white rounded font-semibold min-w-[210px]"
                        disabled={loadingAvanzada || !diagnosticoReal}
                    >
                        {loadingAvanzada ? (
                            <>
                                <svg className="animate-spin h-5 w-5 mr-2 text-white inline-block align-middle" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                                </svg>
                                Calculando...
                            </>
                        ) : (
                            "Calcular similitud"
                        )}
                    </button>

                    {/* Alertas/warnings */}
                    {alerta && (
                        <div className="mt-1 px-4 py-2 bg-yellow-50 border border-yellow-400 text-yellow-900 rounded shadow text-center text-sm w-full">
                            {alerta}
                        </div>
                    )}
                </div>


                {/* Diagnóstico real */}
                <div className="flex-1">
                    <h3 className="font-semibold text-indigo-700 mb-2">Diagnóstico Real</h3>
                    <textarea
                        rows={8}
                        className="w-full border border-indigo-400 rounded-lg p-3 mb-2"
                        value={diagnosticoReal}
                        onChange={e => {
                            setDiagnosticoReal(e.target.value);
                            setSimilitudAvanzada(null);
                        }}
                        placeholder="Pega aquí el diagnóstico escrito por el médico real..."
                    />
                </div>
            </div>
        </section>
    );
}