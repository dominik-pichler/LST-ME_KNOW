FROM --platform=linux/amd64 ubuntu:latest

RUN apt-get update && apt-get install -y git python3 python3-pip

RUN pip install --no-cache-dir --break-system-packages \
    numpy pandas \
    matplotlib seaborn \
    torch torchvision torchaudio torchtext \
    tqdm

CMD ["python3", "./src/autocomplete.py"]
