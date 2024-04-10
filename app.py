import os, json
from flask import Flask, request
from werkzeug.middleware.proxy_fix import ProxyFix
from code.inference import model_fn, predict_fn, input_fn, output_fn


app = Flask(__name__)

# Load the model by reading the `SM_MODEL_DIR` environment variable
# which is passed to the container by SageMaker (usually /opt/ml/model).
model_dir = os.getenv("SM_MODEL_DIR", "opt/ml/model")

# Since the web application runs behind a proxy (nginx), we need to
# add this setting to our app.
app.wsgi_app = ProxyFix(
    app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1
)


@app.route("/ping", methods=["GET"])
def ping():
    """
    Healthcheck function.
    """
    return "pong"


@app.route("/invocations", methods=["POST"])
def invocations():
    """
    Function which responds to the invocations requests.
    """
    payload = json.dumps(request.json)


    response, accept = output_fn(
        predict_fn(
            input_fn(payload, "application/json"),
            model_fn(model_dir)
        ),
        "application/json"
    )

    return response
