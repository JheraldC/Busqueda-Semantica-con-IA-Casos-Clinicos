import React, { useState, useRef } from "react";
import { useNavigate } from "react-router-dom";
import { v4 as uuidv4 } from "uuid";

export default function Formulario({ onSetResultado, onSetError, onSetResultadosHistorial }) {
    const [formData, setFormData] = useState({
        edad: "",
        sexo: "",
        peso: "",
        sintomas: "",
        antecedentes: "",
        texto: "",
    });
    const [tipoConsulta, setTipoConsulta] = useState("buscar");
    const [loading, setLoading] = useState(false);
    const loadingTimeout = useRef(null);
    const navigate = useNavigate();

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        loadingTimeout.current = setTimeout(() => setLoading(true), 900);

        try {
            let url = "";
            let body = {};
            let isImageDiagnosis = tipoConsulta === "diagnostico_imagen";

            if (isImageDiagnosis) {
                url = "http://127.0.0.1:8000/diagnostico_imagen";
            } else if (tipoConsulta === "buscar") {
                url = "http://127.0.0.1:8000/buscar";
                body = {
                    edad: parseInt(formData.edad),
                    sexo: formData.sexo,
                    peso: formData.peso ? parseFloat(formData.peso) : undefined,
                    sintomas: formData.sintomas,
                    antecedentes: formData.antecedentes,
                };
            } else if (tipoConsulta === "buscar_texto") {
                url = "http://127.0.0.1:8000/buscar_texto";
                body = { texto: formData.texto };
            } else if (tipoConsulta === "diagnostico_inteligente") {
                url = "http://127.0.0.1:8000/diagnostico_inteligente";
                body = { texto: formData.texto };
            }

            let response;
            if (isImageDiagnosis) {
                // Enviar imágenes usando FormData
                const form = new FormData();
                if (!formData.imagenes || formData.imagenes.length === 0) {
                    throw new Error("Debe subir al menos una imagen.");
                }
                for (let i = 0; i < formData.imagenes.length; i++) {
                    form.append("files", formData.imagenes[i]);
                }
                response = await fetch(url, {
                    method: "POST",
                    body: form,
                });
            } else {
                response = await fetch(url, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(body),
                });
            }

            if (!response.ok) {
                throw new Error(`Error en consulta: ${response.status} ${response.statusText}`);
            }

            const data = await response.json();
            clearTimeout(loadingTimeout.current);
            setLoading(false);

            const nuevoResultado = {
                id: uuidv4(),
                timestamp: Date.now(),
                tipoConsulta,
                consulta: isImageDiagnosis
                    ? { imagenes: [...formData.imagenes].map(f => f.name) }
                    : body,
                resultado: data,
            };

            // Guardar resultado en el backend
            const saveResponse = await fetch("http://127.0.0.1:8000/guardarResultado", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(nuevoResultado),
            });

            if (!saveResponse.ok) {
                const errorText = await saveResponse.text();
                console.error("Error al guardar resultado:", saveResponse.status, errorText);
                throw new Error(`Error al guardar resultado: ${saveResponse.status}`);
            } else {
                console.log("Resultado guardado exitosamente.");
            }

            // Actualizar historial desde backend
            const resHistorial = await fetch("http://127.0.0.1:8000/resultados");
            if (!resHistorial.ok) {
                throw new Error(`Error al traer historial: ${resHistorial.status}`);
            }
            const historialActualizado = await resHistorial.json();
            if (!Array.isArray(historialActualizado.resultados)) {
                onSetResultadosHistorial([]);
            } else {
                onSetResultadosHistorial(historialActualizado.resultados);
            }

            onSetResultado(nuevoResultado);
            onSetError(null);
            navigate(`/resultado/${nuevoResultado.id}`);

        } catch (err) {
            clearTimeout(loadingTimeout.current);
            setLoading(false);
            console.error("Error general en handleSubmit:", err);
            onSetError(err.message || "Error al consultar el servidor");
            onSetResultado(null);
        }
    };

    return (
        <section className="max-w-3xl mx-auto bg-white p-10 rounded shadow">
            <div className="mb-8">
                <label className="block mb-3 font-semibold text-indigo-900 text-lg">
                    Tipo de Consulta:
                </label>
                <select
                    className="w-full border border-indigo-400 rounded-lg p-3 text-indigo-900 focus:outline-none focus:ring-2 focus:ring-indigo-600"
                    value={tipoConsulta}
                    onChange={(e) => setTipoConsulta(e.target.value)}
                >
                    <option value="buscar">Buscar (Datos Estructurados)</option>
                    <option value="buscar_texto">Buscar Texto Libre</option>
                    <option value="diagnostico_inteligente">Diagnóstico Inteligente</option>
                    <option value="diagnostico_imagen">Diagnóstico por Imágenes Radiológicas</option>
                </select>
            </div>

            <form onSubmit={handleSubmit} className="space-y-6 text-indigo-900">
                {tipoConsulta === "buscar" && (
                    <>
                        <div>
                            <label htmlFor="edad" className="block mb-2 font-medium">
                                Edad <span className="text-red-600">*</span>
                            </label>
                            <input
                                type="number"
                                id="edad"
                                name="edad"
                                placeholder="Ej. 30"
                                className="w-full border border-indigo-400 rounded-lg p-3"
                                value={formData.edad}
                                onChange={handleChange}
                                required
                            />
                        </div>
                        <div>
                            <label htmlFor="sexo" className="block mb-2 font-medium">
                                Sexo <span className="text-red-600">*</span>
                            </label>
                            <input
                                type="text"
                                id="sexo"
                                name="sexo"
                                placeholder="Ej. Masculino / Femenino"
                                className="w-full border border-indigo-400 rounded-lg p-3"
                                value={formData.sexo}
                                onChange={handleChange}
                                required
                            />
                        </div>
                        <div>
                            <label htmlFor="peso" className="block mb-2 font-medium">
                                Peso (opcional)
                            </label>
                            <input
                                type="number"
                                step="0.1"
                                id="peso"
                                name="peso"
                                placeholder="Ej. 70.5"
                                className="w-full border border-indigo-400 rounded-lg p-3"
                                value={formData.peso}
                                onChange={handleChange}
                            />
                        </div>
                        <div>
                            <label htmlFor="sintomas" className="block mb-2 font-medium">
                                Síntomas <span className="text-red-600">*</span>
                            </label>
                            <textarea
                                id="sintomas"
                                name="sintomas"
                                placeholder="Describe los síntomas"
                                className="w-full border border-indigo-400 rounded-lg p-3"
                                rows="4"
                                value={formData.sintomas}
                                onChange={handleChange}
                                required
                            ></textarea>
                        </div>
                        <div>
                            <label htmlFor="antecedentes" className="block mb-2 font-medium">
                                Antecedentes (opcional)
                            </label>
                            <textarea
                                id="antecedentes"
                                name="antecedentes"
                                placeholder="Describe antecedentes familiares o personales"
                                className="w-full border border-indigo-400 rounded-lg p-3"
                                rows="3"
                                value={formData.antecedentes}
                                onChange={handleChange}
                            ></textarea>
                        </div>
                    </>
                )}

                {(tipoConsulta === "buscar_texto" || tipoConsulta === "diagnostico_inteligente") && (
                    <div>
                        <label htmlFor="texto" className="block mb-2 font-medium">
                            Consulta en texto libre <span className="text-red-600">*</span>
                        </label>
                        <textarea
                            id="texto"
                            name="texto"
                            placeholder="Describe tu consulta médica aquí"
                            className="w-full border border-indigo-400 rounded-lg p-3"
                            rows="6"
                            value={formData.texto}
                            onChange={handleChange}
                            required
                        ></textarea>
                    </div>
                )}

                {tipoConsulta === "diagnostico_imagen" && (
                    <div>
                        <label htmlFor="imagenes" className="block mb-2 font-medium">
                            Subir imágenes (máx. 3, formatos: jpg, png)
                        </label>
                        <input
                            type="file"
                            id="imagenes"
                            name="imagenes"
                            accept="image/*"
                            multiple
                            className="w-full border border-indigo-400 rounded-lg p-3"
                            onChange={(e) => setFormData({ ...formData, imagenes: e.target.files })}
                            required
                        />
                    </div>
                )}


                {loading && (
                    <div className="flex justify-center mb-4">
                        <svg className="animate-spin h-6 w-6 text-indigo-600" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle
                                className="opacity-25"
                                cx="12" cy="12" r="10"
                                stroke="currentColor" strokeWidth="4"
                            ></circle>
                            <path
                                className="opacity-75"
                                fill="currentColor"
                                d="M4 12a8 8 0 018-8v8H4z"
                            ></path>
                        </svg>
                    </div>
                )}

                <button
                    type="submit"
                    className="w-full bg-indigo-600 hover:bg-indigo-800 text-white font-semibold py-3 rounded-lg transition duration-200"
                >
                    Enviar Consulta
                </button>
            </form>
        </section>
    );
}
