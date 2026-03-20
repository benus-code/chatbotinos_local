FROM rasa/rasa-sdk:3.6.2

WORKDIR /app

# FR: Installer PyTorch CPU-only en premier pour éviter le téléchargement des libs CUDA (~4 GB).
# RU: Установить PyTorch CPU-only первым, чтобы избежать загрузки CUDA-библиотек (~4 ГБ).
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    qdrant-client \
    sentence-transformers \
    requests \
    pypdf

COPY actions/ /app/actions/

EXPOSE 5055

CMD ["start", "--actions", "actions", "--port", "5055"]
