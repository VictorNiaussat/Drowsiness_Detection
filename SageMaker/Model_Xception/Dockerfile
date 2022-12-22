FROM tensorflow/tensorflow:latest-gpu

ENV PYTHONUNBUFFERED=True

ENV CLOUD_PATH="/opt/ml/code"

COPY code/ $CLOUD_PATH

RUN pip install -r /opt/ml/code/requirements.txt

# make the entrypoint script executable
RUN chmod +x $CLOUD_PATH/entrypoint.sh

WORKDIR $CLOUD_PATH
ENTRYPOINT ["./entrypoint.sh"]