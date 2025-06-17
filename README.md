# sturdy-telegram
Apache Airflow Playground

<p align="center">
    <img src="images/airflow_attempt1.png">
</p>

This GitHub repository is a companion to the Medium article [How to build your own change data capture pipeline using Apache Airflow](https://medium.com/mitb-for-all/how-to-build-your-own-change-data-capture-pipeline-using-apache-airflow-e485fbef82c7)

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
