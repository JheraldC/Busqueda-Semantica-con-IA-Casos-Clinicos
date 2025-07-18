# Obtener la carpeta actual donde se ejecuta el script
$basePath = (Get-Location).Path

# Ejecutar frontend
Write-Host "Iniciando frontend..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd `"$basePath\frontend`"; npm start"

# Ejecutar backend (activar venv y uvicorn)
Write-Host "Iniciando backend..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd `"$basePath\backend`"; .\venv\Scripts\Activate.ps1; uvicorn main:app --reload"

# Ejecutar ollama serve
Write-Host "Iniciando ollama serve..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd `"$basePath`"; ollama serve"

Write-Host "Todos los servidores están iniciados en nuevas ventanas de PowerShell."
