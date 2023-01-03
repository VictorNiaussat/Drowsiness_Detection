FROM tensorflow/tensorflow:latest-gpu

ENV PYTHONUNBUFFERED=True

ENV CLOUD_PATH="/opt/ml/code"
ADD requirements.txt /
RUN pip install -r /requirements.txt

COPY . $CLOUD_PATH



# make the entrypoint script executable
RUN chmod +x $CLOUD_PATH/entrypoint.sh

WORKDIR $CLOUD_PATH
ENTRYPOINT ["./entrypoint.sh"]