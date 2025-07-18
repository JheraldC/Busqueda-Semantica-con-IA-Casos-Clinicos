import React, { useState, useEffect } from "react";
import { BrowserRouter as Router, Routes, Route, NavLink } from "react-router-dom";
import Formulario from "./Formulario";
import Inicio from "./Inicio";
import Resultado from "./Resultado";
import { FaStethoscope } from "react-icons/fa";

function App() {
  const [resultadosHistorial, setResultadosHistorial] = useState([]);
  const [resultadoActual, setResultadoActual] = useState(null);
  const [error, setError] = useState(null);

  // Cargar historial desde backend al iniciar la app
  useEffect(() => {
    fetch("http://127.0.0.1:8000/resultados")
      .then(res => res.json())
      .then(data => {
        if (data.resultados) {
          setResultadosHistorial(data.resultados);
        }
      })
      .catch(() => setError("Error al cargar resultados desde backend"));
  }, []);

  return (
    <Router>
      <header className="bg-indigo-700 p-4 shadow-md flex items-center justify-between text-indigo-100">
        <h1 className="text-2xl font-bold flex items-center space-x-2">
          <FaStethoscope className="text-indigo-300 text-3xl" />
          <span>Consulta Médica</span>
        </h1>
        <nav className="space-x-6 font-semibold">
          <NavLink
            to="/"
            end
            className={({ isActive }) =>
              isActive ? "text-indigo-300 underline" : "hover:text-indigo-300"
            }
          >
            Inicio
          </NavLink>
          <NavLink
            to="/formulario"
            className={({ isActive }) =>
              isActive ? "text-indigo-300 underline" : "hover:text-indigo-300"
            }
          >
            Formulario
          </NavLink>
          <NavLink
            to="/resultado"
            className={({ isActive }) =>
              isActive ? "text-indigo-300 underline" : "hover:text-indigo-300"
            }
          >
            Resultado
          </NavLink>
        </nav>
      </header>

      <main className="min-h-[calc(100vh-64px)] bg-gray-50 p-6">
        <Routes>
          <Route path="/" element={<Inicio />} />
          <Route
            path="/formulario"
            element={
              <Formulario
                onSetResultado={setResultadoActual}
                onSetError={setError}
                onSetResultadosHistorial={setResultadosHistorial} // Solo actualiza desde backend
              />
            }
          />
          <Route
            path="/resultado"
            element={
              <Resultado
                resultadosHistorial={resultadosHistorial}
                error={error}
                onSetResultadosHistorial={setResultadosHistorial}
              />
            }
          />
          <Route
            path="/resultado/:id"
            element={
              <Resultado
                resultadosHistorial={resultadosHistorial}
                error={error}
                onSetResultadosHistorial={setResultadosHistorial}
              />
            }
          />
        </Routes>
      </main>

      <footer className="bg-indigo-800 p-4 text-center text-indigo-200 mt-auto border-t border-indigo-700">
        <p>© 2025 Proyecto IA Médica - Tacna, Perú</p>
      </footer>
    </Router>
  );
}

export default App;