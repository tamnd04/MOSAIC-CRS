$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

python realtime_voice_app.py `
  --config config_redial.yaml `
  --checkpoint checkpoints\ReDial\best_rl_model.pt `
  --train-data data\ReDial\train_data.json `
  --catalog data\ReDial\item_catalog.json `
  --no-auto-catalog `
  --no-ollama `
  --host 127.0.0.1 `
  --port 7860
