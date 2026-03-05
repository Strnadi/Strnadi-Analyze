# Dockerfile
FROM vastai/base-image:cuda-12.8.1-auto

# install python deps into the main venv
RUN . /venv/main/bin/activate && \
    pip install --no-cache-dir torch torchvision fastapi uvicorn[standard] pillow

# copy model server and start script
COPY model_server.py /opt/model_server.py
COPY start.sh /opt/start.sh
RUN chmod +x /opt/start.sh

# expose the model-server port used by the pyworker (choose 8080)
EXPOSE 8080

ENTRYPOINT ["/opt/start.sh"]
