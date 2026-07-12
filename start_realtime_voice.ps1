$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

if (-not (Test-Path ".env")) {
    Copy-Item ".env.realtime.example" ".env"
    Write-Host "Created .env with local defaults. No API key is needed."
}

Write-Host "Checking Ollama..."
try {
    $null = Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 2
} catch {
    Write-Warning "Ollama is not reachable. The app will use deterministic response templates."
    Write-Warning "Install/start Ollama and run: ollama pull llama3.2:3b"
}

python realtime_voice_app.py `
  --config config_redial.yaml `
  --checkpoint checkpoints\ReDial\best_rl_model.pt `
  --train-data data\ReDial\train_data.json `
  --catalog data\ReDial\item_catalog.json `
  --no-auto-catalog `
  --host 127.0.0.1 `
  --port 7860
