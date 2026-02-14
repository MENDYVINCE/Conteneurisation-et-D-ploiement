FROM python:3.12-slim

WORKDIR /app

# Installer dépendances système
RUN apt update && apt install -y git curl build-essential && rm -rf /var/lib/apt/lists/*

# Installer Poetry
RUN pip install poetry

# Copier les fichiers Poetry
COPY pyproject.toml poetry.lock* /app/

# Installer les dépendances
RUN poetry config virtualenvs.create false \
    && poetry install --with visu,dev --no-root --no-ansi --no-directory


# Copier le code et données
COPY artifacts/ /app/artifacts/
COPY src/ /app/src/
COPY dev/app.py /app/
COPY dev/front.py /app/
COPY dev/predictor.py /app/
COPY datas/ /app/datas/
COPY dev/start.sh /app/start.sh

# Ajouter le PATH pour Poetry/local bin
ENV PATH="/root/.local/bin:$PATH"


# Exposer les ports
EXPOSE 5000 8501


RUN chmod +x /app/start.sh
CMD ["/app/start.sh"]
