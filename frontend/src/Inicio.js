import React from "react";

export default function Inicio() {
  return (
    <section className="max-w-4xl mx-auto mt-16 p-10 bg-white rounded shadow text-purple-900 text-center">
      <h2 className="text-3xl font-bold mb-6">Bienvenido a Consulta Médica IA</h2>
      <p className="mb-6 text-lg">
        Esta plataforma permite consultar casos médicos usando inteligencia artificial basada en casos reales.
      </p>
      <p className="mb-4 text-gray-700 max-w-2xl mx-auto text-left">
        Aquí puedes realizar tres tipos de consultas para obtener una orientación médica precisa y personalizada:
      </p>

      <div className="text-left max-w-2xl mx-auto space-y-4">
        <div>
          <h3 className="font-semibold text-purple-800 mb-1">Buscar (Datos Estructurados)</h3>
          <p className="text-gray-700">
            Ingresa datos específicos y estructurados como edad, sexo, síntomas y antecedentes para encontrar casos clínicos similares basados en información detallada y organizada.
          </p>
        </div>

        <div>
          <h3 className="font-semibold text-purple-800 mb-1">Buscar Texto Libre</h3>
          <p className="text-gray-700">
            Describe tu consulta o síntomas en un texto libre sin formato específico. La IA analizará el texto y buscará casos clínicos que coincidan con la descripción que proporciones.
          </p>
        </div>

        <div>
          <h3 className="font-semibold text-purple-800 mb-1">Diagnóstico Inteligente</h3>
          <p className="text-gray-700">
            Basado en los casos clínicos más similares encontrados, el sistema generará un diagnóstico probable, una breve explicación, especialidad médica recomendada y preguntas clave para la consulta médica.
          </p>
        </div>

        <div className="mb-4">
          <h3 className="font-semibold text-purple-800 mb-1">Diagnóstico por Imágenes Médicas</h3>
          <p className="text-gray-700">
            El sistema analiza imágenes radiológicas usando inteligencia artificial para generar un diagnóstico probable,
            acompañado de una explicación médica, la especialidad sugerida y preguntas clínicas relevantes para apoyar la toma de decisiones médicas.
          </p>
        </div>

      </div>

      <p className="mt-8 text-gray-700 max-w-2xl mx-auto">
        Navega a la pestaña <strong>Formulario</strong> para comenzar tu consulta médica con cualquiera de estos métodos.
      </p>
    </section>
  );
}
