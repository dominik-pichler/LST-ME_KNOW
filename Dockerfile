FROM ubuntu:20.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s /root/.local/bin/poetry /usr/local/bin/poetry

# Set the working directory
WORKDIR /app

# Copy Poetry files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --no-root

# Copy the rest of the application code
COPY . .

# Specify the command to run the application
CMD ["poetry", "run", "python3", "./src/lstm_auto_complete.py"]
