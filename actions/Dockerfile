FROM rasa/rasa-sdk:3.6.2

WORKDIR /app

# FR: rasa-sdk tourne en user non-root — on repasse root pour les installs pip.
# RU: rasa-sdk запускается от non-root — переключаемся на root для установки pip.
USER root

# FR: Installer PyTorch CPU-only en premier pour éviter le téléchargement des libs CUDA (~4 GB).
# RU: Установить PyTorch CPU-only первым, чтобы избежать загрузки CUDA-библиотек (~4 ГБ).
RUN /opt/venv/bin/pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

RUN /opt/venv/bin/pip install --no-cache-dir \
    qdrant-client \
    sentence-transformers \
    requests \
    pypdf

# FR: Repasser sur l'utilisateur non-root de rasa-sdk pour l'exécution.
# RU: Вернуться к non-root пользователю rasa-sdk для запуска.
USER 1001

# FR: Ajouter /app/actions au PYTHONPATH pour que `from indexation import ...` fonctionne.
# RU: Добавить /app/actions в PYTHONPATH, чтобы `from indexation import ...` работал.
ENV PYTHONPATH=/app/actions

COPY actions/ /app/actions/

EXPOSE 5055

CMD ["start", "--actions", "actions", "--port", "5055"]
