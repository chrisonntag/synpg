FROM 763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-inference:1.2.0-gpu-py36-cu100-ubuntu16.04

# Set environment variables
# ensures that Python outputs everything that's printed directly to the terminal (so logs can be seen in real-time)
ENV PYTHONUNBUFFERED=TRUE
# ensures Python doesn't try to write .pyc files to disk (useful for improving performance in some scenarios)
ENV PYTHONDONTWRITEBYTECODE=TRUE
# Update PATH environment variable to include /opt/program directory
ENV PATH="/opt/ml/code:${PATH}"

WORKDIR /opt/ml/code

COPY ./requirements.txt /opt/ml/code/requirements.txt

COPY ./app.py /opt/ml/code/app.py
COPY ./code/ /opt/ml/code/
COPY ./synpg /opt/ml/code/synpg/

RUN ls -laR /opt/ml/code/*

ENV SM_MODEL_DIR /opt/ml/model
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
ENV SAGEMAKER_PROGRAM inference.py

RUN pip install --no-cache-dir "git+https://github.com/chrisonntag/synpg.git"
RUN pip install --no-cache-dir -r requirements.txt
RUN pip freeze
