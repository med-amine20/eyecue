FROM python:3.9

# Éviter les erreurs d'installation
ENV DEBIAN_FRONTEND=noninteractive

# Installer Java et autres dépendances
RUN apt-get update && \
    apt-get install -y openjdk-11-jdk curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Définir JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

# Installer les dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code backend
COPY . /app
WORKDIR /app

CMD ["python", "app.py"]
