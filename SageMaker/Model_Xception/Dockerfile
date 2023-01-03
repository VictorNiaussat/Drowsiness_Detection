FROM tensorflow/tensorflow:latest-gpu

ENV PYTHONUNBUFFERED=True

ENV CLOUD_PATH="/opt/ml/code"

RUN pip install -r /requirements.txt

COPY . $CLOUD_PATH

RUN pip install -r /opt/ml/program/requirements.txt


# make the entrypoint script executable
RUN chmod +x $CLOUD_PATH/entrypoint.sh

WORKDIR $CLOUD_PATH
ENTRYPOINT ["./entrypoint.sh"]