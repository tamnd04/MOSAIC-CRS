#!/usr/bin/env sh
set -eu

if [ ! -f .env ]; then
  cp .env.realtime.example .env
  echo "Created .env. Add OPENAI_API_KEY, then run this script again."
  exit 1
fi

python realtime_voice_app.py \
  --config config_redial.yaml \
  --checkpoint checkpoints\\ReDial\\best_rl_model.pt \
  --train-data data\\ReDial\\train_data.json \
  --host 127.0.0.1 \
  --port 7860
