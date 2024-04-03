FROM 763104351884.dkr.ecr.eu-central-1.amazonaws.com/pytorch-inference:1.2.0-gpu-py36-cu100-ubuntu16.04

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir "git+https://github.com/chrisonntag/synpg.git"
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV SM_MODEL_DIR /opt/ml/model

ENTRYPOINT ["gunicorn", "-b", "0.0.0.0:8080", "app:app", "--timeout", "120", "-n"]
