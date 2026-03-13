FROM rasa/rasa-sdk:3.6.2

WORKDIR /app

RUN pip install --no-cache-dir \
    qdrant-client \
    sentence-transformers \
    requests \
    pypdf \
    langchain

COPY actions/ /app/actions/

EXPOSE 5055

CMD ["start", "--actions", "actions", "--port", "5055"]
