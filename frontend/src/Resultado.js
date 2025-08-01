import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { Link } from "react-router-dom";
import { FiChevronDown, FiChevronRight } from "react-icons/fi";

export default function Resultado({
    resultadosHistorial,
    onSetResultadosHistorial,
    resultadoActualProp,
    error,
}) {
    const { id } = useParams();
    const [loading, setLoading] = useState(false);
    const [resultadoActual, setResultadoActual] = useState(null);
    const [expandidos, setExpandidos] = useState({}); // Para controlar expandido por ID
    const [cargandoIA, setCargandoIA] = useState(false);
    const [resultadoIA, setResultadoIA] = useState({}); // Guarda respuestas IA por consulta ID

    useEffect(() => {
        if (resultadoActualProp) {
            setResultadoActual(resultadoActualProp);
            return;
        }
        if (id && resultadosHistorial && resultadosHistorial.length > 0) {
            const encontrado = resultadosHistorial.find((r) => r?.id === id);
            setResultadoActual(encontrado || null);
            return;
        }
        if (!resultadosHistorial || resultadosHistorial.length === 0) {
            async function fetchHistorial() {
                setLoading(true);
                try {
                    const res = await fetch("http://127.0.0.1:8000/resultados");
                    const data = await res.json();
                    if (data.resultados) {
                        onSetResultadosHistorial(data.resultados);
                        if (id) {
                            const encontrado = data.resultados.find((r) => r?.id === id);
                            setResultadoActual(encontrado || null);
                        }
                    }
                } catch (err) {
                    console.error("Error al cargar resultados", err);
                } finally {
                    setLoading(false);
                }
            }
            fetchHistorial();
        }
    }, [id, resultadosHistorial, onSetResultadosHistorial, resultadoActualProp]);

    const toggleExpand = (id) => {
        setExpandidos((prev) => ({ ...prev, [id]: !prev[id] }));
    };

    // Genera texto para IA a partir de un resultado
    const generarTextoParaIA = (resultado) => {
        if (!resultado) return "";
        let texto = `Consulta médica con ID ${resultado.id}. Fecha: ${new Date(resultado.timestamp).toLocaleString()}.\n`;
        texto += `Tipo de consulta: ${resultado.tipoConsulta.replace(/_/g, " ")}.\n`;

        if (resultado.consulta) {
            if (resultado.tipoConsulta === "buscar") {
                texto += `Edad: ${resultado.consulta.edad || "N/A"}, Sexo: ${resultado.consulta.sexo || "N/A"}, Peso: ${resultado.consulta.peso || "N/A"}.\n`;
                texto += `Síntomas: ${resultado.consulta.sintomas || "N/A"}.\n`;
                texto += `Antecedentes: ${resultado.consulta.antecedentes || "N/A"}.\n`;
            } else {
                texto += `Texto de consulta: ${resultado.consulta.texto || "N/A"}.\n`;
            }
        }

        if (resultado.resultado?.resultados && resultado.resultado.resultados.length > 0) {
            texto += `Resultados:\n`;
            resultado.resultado.resultados.forEach((item, i) => {
                texto += `Resultado ${i + 1} - Motivo: ${item.motivo_consulta || "N/A"}, Diagnóstico: ${item.diagnostico || "N/A"}.\n`;
                if (item.historia_actual) texto += `Historia actual: ${item.historia_actual}.\n`;
                if (item.antecedentes) texto += `Antecedentes: ${item.antecedentes}.\n`;
            });
        }
        return texto;
    };

    // Función para consultar IA por diagnóstico personalizado
    const consultarDiagnosticoIA = async (resultado) => {
        setCargandoIA(true);
        setResultadoIA((prev) => ({ ...prev, [resultado.id]: "Consultando... Por favor, espere." }));
        try {
            const textoConsulta = generarTextoParaIA(resultado);
            const res = await fetch("http://127.0.0.1:8000/diagnostico_inteligente", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ texto: textoConsulta }),
            });
            const data = await res.json();
            setResultadoIA((prev) => ({
                ...prev,
                [resultado.id]: data.diagnostico_probable ? JSON.stringify(data) : (data.diagnostico || JSON.stringify(data) || "Sin respuesta de IA"),
            }));
        } catch (error) {
            setResultadoIA((prev) => ({ ...prev, [resultado.id]: "Error al consultar IA" }));
            console.error(error);
        } finally {
            setCargandoIA(false);
        }
    };

    if (loading) {
        return <p>Cargando resultados...</p>;
    }

    if (error) {
        return (
            <div className="max-w-4xl mx-auto p-6 bg-red-100 text-red-700 rounded shadow">
                <h2 className="text-2xl font-semibold mb-4">Error</h2>
                <p>{error}</p>
            </div>
        );
    }

    if (id && !resultadoActual) {
        return (
            <div className="max-w-4xl mx-auto p-6 bg-yellow-100 text-yellow-700 rounded shadow">
                <h2 className="text-2xl font-semibold mb-4">Resultado no encontrado.</h2>
                <p>No existe un resultado para el ID proporcionado.</p>
            </div>
        );
    }

    // Render cuando hay id y resultadoActual
    if (id && resultadoActual) {
        // Detectar si es diagnóstico inteligente por estructura (ej: sin .resultados pero con .diagnostico_probable)
        const esDiagnosticoInteligente = !!resultadoActual.resultado?.diagnostico_probable;

        // Para resultados tipo lista
        const lista = resultadoActual.resultado?.resultados || [];

        return (
            <section className="max-w-4xl mx-auto p-6 bg-white rounded shadow text-indigo-900">
                <h2 className="text-2xl font-semibold mb-6">Resultados para consulta ID: {id}</h2>

                {esDiagnosticoInteligente ? (
                    <div className="p-4 bg-indigo-100 rounded border border-indigo-300 text-indigo-900 whitespace-pre-wrap">
                        <p><strong>Diagnóstico probable:</strong> {resultadoActual.resultado.diagnostico_probable}</p>
                        <p><strong>Explicación:</strong> {resultadoActual.resultado.explicacion}</p>
                        <p><strong>Especialidad recomendada:</strong> {resultadoActual.resultado.especialidad_recomendada}</p>
                        <p><strong>Preguntas para consulta:</strong> {resultadoActual.resultado.preguntas_consulta}</p>
                    </div>
                ) : lista.length === 0 ? (
                    <>
                        <p>No se encontraron resultados.</p>
                        {resultadoActual.tipoConsulta !== "diagnostico_inteligente" && (
                            <button
                                disabled={cargandoIA}
                                onClick={() => consultarDiagnosticoIA(resultadoActual)}
                                className="mt-6 px-6 py-3 bg-indigo-600 hover:bg-indigo-700 rounded text-white font-semibold"
                            >
                                {cargandoIA ? "Consultando IA..." : "Consulta Diagnóstico IA"}
                            </button>
                        )}
                        {resultadoIA[resultadoActual.id] && (
                            <div className="mt-4 p-4 bg-indigo-100 rounded border border-indigo-300 text-indigo-900 whitespace-pre-wrap">
                                {(() => {
                                    let json;
                                    try {
                                        json = JSON.parse(resultadoIA[resultadoActual.id]);
                                    } catch {
                                        json = null;
                                    }
                                    if (!json) {
                                        return (
                                            <>
                                                <strong>Respuesta IA:</strong>
                                                <p>{resultadoIA[resultadoActual.id]}</p>
                                            </>
                                        );
                                    }
                                    return (
                                        <>
                                            <strong>Respuesta IA:</strong>
                                            <p><strong>Diagnóstico probable:</strong> {json.diagnostico_probable}</p>
                                            <p><strong>Explicación:</strong> {json.explicacion}</p>
                                            <p><strong>Especialidad recomendada:</strong> {json.especialidad_recomendada}</p>
                                            <p><strong>Preguntas para consulta:</strong> {json.preguntas_consulta}</p>
                                        </>
                                    );
                                })()}
                            </div>
                        )}
                    </>
                ) : (
                    lista.map((item, index) => (
                        <div
                            key={index}
                            className="mb-4 p-4 bg-indigo-50 rounded shadow border border-indigo-200"
                        >
                            <p>
                                <strong className="text-indigo-700">Motivo:</strong>{" "}
                                <span className="text-indigo-900">{item.motivo_consulta}</span>
                            </p>
                            <p>
                                <strong className="text-indigo-700">Diagnóstico:</strong>{" "}
                                <span className="text-indigo-800 font-semibold">{item.diagnostico}</span>
                            </p>
                            {item.historia_actual && (
                                <p>
                                    <strong className="text-indigo-700">Historia actual:</strong>{" "}
                                    <span className="italic text-indigo-600">{item.historia_actual}</span>
                                </p>
                            )}
                            {item.antecedentes && (
                                <p>
                                    <strong className="text-indigo-700">Antecedentes:</strong>{" "}
                                    <span className="text-indigo-600">{item.antecedentes}</span>
                                </p>
                            )}
                        </div>
                    ))
                )}
            </section>
        );
    }

    if (resultadosHistorial.length === 0) {
        return (
            <div className="max-w-4xl mx-auto p-6 bg-yellow-100 text-yellow-700 rounded shadow">
                <h2 className="text-2xl font-semibold mb-4">No hay resultados para mostrar.</h2>
            </div>
        );
    }

    // Render para lista de resultados
    return (
        <section className="max-w-4xl mx-auto p-6 bg-white rounded shadow text-indigo-900">
            <h2 className="text-2xl font-semibold mb-6">Últimos Resultados</h2>
            {resultadosHistorial
                .filter(Boolean)
                .slice(-5)
                .reverse()
                .map((resultado) => {
                    const fecha = new Date(resultado.timestamp).toLocaleString(); // Formato local
                    const esDiagnosticoInteligente = !!resultado.resultado?.diagnostico_probable;
                    const esDiagnosticoImagen = resultado?.tipoConsulta === "diagnostico_imagen";

                    return (
                        <div
                            key={resultado.id}
                            className="mb-4 border-b border-indigo-200 pb-2"
                        >
                            <button
                                className="flex items-center space-x-2 text-indigo-800 hover:text-indigo-600 focus:outline-none"
                                onClick={() => toggleExpand(resultado.id)}
                                aria-expanded={!!expandidos[resultado.id]}
                                aria-controls={`detalle-${resultado.id}`}
                            >
                                <span className="font-semibold">Consulta ID:</span>
                                <span>{resultado.id}</span>
                                <span className="ml-4 text-sm text-gray-500">
                                    Fecha: {fecha}
                                </span>
                                {resultado.tipoConsulta && (
                                    <span className="ml-4 px-2 py-0.5 text-xs bg-indigo-200 rounded text-indigo-800">
                                        {resultado.tipoConsulta.replace("_", " ")}
                                    </span>
                                )}
                                {expandidos[resultado.id] ? (
                                    <FiChevronDown className="transition-transform duration-300" />
                                ) : (
                                    <FiChevronRight className="transition-transform duration-300" />
                                )}
                            </button>

                            {expandidos[resultado.id] && (
                                <div
                                    id={`detalle-${resultado.id}`}
                                    className="mt-2 pl-6 space-y-4"
                                >
                                    {/* Mostrar contenido según tipo */}
                                    {esDiagnosticoInteligente ? (
                                        <div className="p-4 bg-indigo-100 rounded border border-indigo-300 text-indigo-900 whitespace-pre-wrap">
                                            <p><strong>Diagnóstico probable:</strong> {resultado.resultado.diagnostico_probable}</p>
                                            <p><strong>Explicación:</strong> {resultado.resultado.explicacion}</p>
                                            <p><strong>Especialidad recomendada:</strong> {resultado.resultado.especialidad_recomendada}</p>
                                            <p><strong>Preguntas para consulta:</strong> {resultado.resultado.preguntas_consulta}</p>
                                        </div>
                                    ) : (
                                        (resultado.resultado?.resultados || [])
                                            .filter(Boolean)
                                            .map((item, idx) =>
                                                esDiagnosticoImagen ? (
                                                    <div key={idx} className="p-4 bg-indigo-50 rounded shadow-sm border border-indigo-200">
                                                        <div className="text-sm text-indigo-700 mb-1">
                                                            Imagen: {item.imagen}
                                                        </div>
                                                        {item.descripcion?.hallazgos_principales && (
                                                            <div>
                                                                <strong>Hallazgos principales:</strong> {item.descripcion.hallazgos_principales}
                                                            </div>
                                                        )}
                                                        {item.descripcion?.diagnostico_probable && (
                                                            <div>
                                                                <strong>Diagnóstico probable:</strong> {item.descripcion.diagnostico_probable}
                                                            </div>
                                                        )}
                                                    </div>
                                                ) : (
                                                    <div key={idx} className="p-4 bg-indigo-50 rounded shadow-sm border border-indigo-200">
                                                        <p>
                                                            <strong className="text-indigo-700">Motivo:</strong>{" "}
                                                            <span className="text-indigo-900">{item.motivo_consulta}</span>
                                                        </p>
                                                        <p>
                                                            <strong className="text-indigo-700">Diagnóstico:</strong>{" "}
                                                            <span className="text-indigo-800 font-semibold">{item.diagnostico}</span>
                                                        </p>
                                                        {item.historia_actual && (
                                                            <p>
                                                                <strong className="text-indigo-700">Historia actual:</strong>{" "}
                                                                <span className="italic text-indigo-600">{item.historia_actual}</span>
                                                            </p>
                                                        )}
                                                        {item.antecedentes && (
                                                            <p>
                                                                <strong className="text-indigo-700">Antecedentes:</strong>{" "}
                                                                <span className="text-indigo-600">{item.antecedentes}</span>
                                                            </p>
                                                        )}
                                                    </div>
                                                )
                                            )

                                    )}
                                    {/* Mostrar Hallazgo final y Diagnóstico global solo si es diagnostico_imagen */}
                                    {resultado.tipoConsulta === "diagnostico_imagen" && (
                                        <div className="mt-4 p-4 bg-green-50 rounded shadow border border-green-200">
                                            <div>
                                                <strong>Hallazgo final:</strong>
                                                <div>{resultado.resultado?.hallazgo_final || "No disponible"}</div>
                                            </div>
                                            <div className="mt-2">
                                                <strong>Diagnóstico global:</strong>
                                                <div>{resultado.resultado?.diagnostico_global || "No disponible"}</div>
                                            </div>
                                        </div>
                                    )}

                                    {/* Botón IA solo para resultados tipo lista, no diagnóstico inteligente */}
                                    {!esDiagnosticoInteligente && resultado.tipoConsulta !== "diagnostico_imagen" && (
                                        <>
                                            <button
                                                disabled={cargandoIA}
                                                onClick={() => consultarDiagnosticoIA(resultado)}
                                                className="mt-4 px-4 py-2 bg-indigo-600 hover:bg-indigo-700 rounded text-white font-semibold"
                                            >
                                                {cargandoIA ? "Consultando..." : "Consulta Diagnóstico IA"}
                                            </button>
                                            {resultadoIA[resultado.id] && (
                                                (() => {
                                                    let json;
                                                    try {
                                                        json = JSON.parse(resultadoIA[resultado.id]);
                                                    } catch {
                                                        json = null;
                                                    }
                                                    if (!json) {
                                                        return (
                                                            <div className="mt-2 p-3 bg-indigo-100 rounded border border-indigo-300 text-indigo-900 whitespace-pre-wrap">
                                                                <strong>Respuesta IA:</strong>
                                                                <p>{resultadoIA[resultado.id]}</p>
                                                            </div>
                                                        );
                                                    }
                                                    return (
                                                        <div className="mt-2 p-3 bg-indigo-100 rounded border border-indigo-300 text-indigo-900 whitespace-pre-wrap">
                                                            <strong>Respuesta IA:</strong>
                                                            <p><strong>Diagnóstico probable:</strong> {json.diagnostico_probable}</p>
                                                            <p><strong>Explicación:</strong> {json.explicacion}</p>
                                                            <p><strong>Especialidad recomendada:</strong> {json.especialidad_recomendada}</p>
                                                            <p><strong>Preguntas para consulta:</strong> {json.preguntas_consulta}</p>
                                                        </div>
                                                    );
                                                })()
                                            )}
                                        </>
                                    )}
                                    {resultado.tipoConsulta === "diagnostico_imagen" && (
                                        <div className="mt-6">
                                            <Link
                                                to={`/resultado-imagen/${resultado.id}`}
                                                className="inline-block px-4 py-2 bg-indigo-700 hover:bg-indigo-900 text-white rounded font-semibold transition"
                                            >
                                                Comparar diagnóstico IA vs diagnóstico Real
                                            </Link>

                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    );
                })}
        </section>
    );
}
