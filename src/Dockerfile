FROM python:3.12-slim

WORKDIR /app

RUN apt update && apt install -y git curl build-essential && rm -rf /var/lib/apt/lists/*

RUN pip install poetry

# Copier les fichiers Poetry
COPY pyproject.toml poetry.lock* ./

# Installer les dépendances
RUN poetry config virtualenvs.create false && poetry install --with dev --no-root --no-ansi --no-directory

# Créer dossier pour artefacts
RUN mkdir -p artifacts

# Copier le reste des fichiers (code et données)
COPY datas/ai4i2020.csv datas/
COPY src/ /app/src/

ENV PATH="/root/.local/bin:$PATH"

VOLUME ["/app/artifacts"]

CMD ["python", "src/train.py"]


