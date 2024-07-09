# imagem base do Python
FROM python:3.9-slim

# define o diretório de trabalho
WORKDIR /app

# copia apenas os diretórios e arquivos necessários
COPY model_training /app/model_training
COPY requirements.txt /app/requirements.txt
COPY src /app/src
COPY streamlit /app/streamlit

# instala os pacotes necessários
RUN pip install --no-cache-dir -r requirements.txt

# define o PYTHONPATH
ENV PYTHONPATH="${PYTHONPATH}:/app:/app/src"

# expõe a porta que o Streamlit usará
EXPOSE 8501

# comando para rodar o main.py e depois o Streamlit
CMD ["sh", "-c", "python model_training/main.py && streamlit run streamlit/app.py --server.port=8501 --server.address=0.0.0.0"]
