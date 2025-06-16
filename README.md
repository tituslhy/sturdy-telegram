# sturdy-telegram
Apache Airflow Playground

## Setup
This repository uses the [uv package installer](https://docs.astral.sh/uv/pip/packages/). 

To create a virtual environment with the dependencies installed, simply type in your terminal:
```
uv sync
```

Ollama is also needed. Visit [Ollama's Download Documentation](https://ollama.com/download) to install Ollama on your machine.

Once done, run
```
ollama pull nomic-embed-text
```