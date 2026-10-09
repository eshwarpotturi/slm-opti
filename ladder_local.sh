#!/bin/bash
# Runs the whole ladder for one small model on this Mac through Ollama. Free; nothing leaves the computer.
# Usage:  bash ladder_local.sh llama3.2:3b
M=${1:-llama3.2:3b}
cd "$(dirname "$0")"
ollama pull all-minilm >/dev/null 2>&1
python3 run_eval.py "ollama:$M" baseline
for L in 1 12 123 1234; do python3 stack.py "ollama:$M" --layers $L --tag local; done
