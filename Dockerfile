FROM rasa/rasa-sdk:3.6.2

WORKDIR /app

USER root

COPY requirements-actions.txt /tmp/requirements-actions.txt

RUN pip install --no-cache-dir --timeout 300 --retries 5 --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --timeout 300 --retries 5 --index-url https://download.pytorch.org/whl/cpu torch==2.1.2 \
    && pip install --no-cache-dir --timeout 300 --retries 5 -r /tmp/requirements-actions.txt \
    && rm -f /tmp/requirements-actions.txt

COPY actions /app/actions

USER 1001
