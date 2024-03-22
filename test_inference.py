import json
from inference import model_fn, predict_fn, input_fn, output_fn

payload = '{\
        "sent": "we will have a picnic if it is a sunny day tomorrow.",\
        "synt": "(ROOT (S (NP (PRP we)) (VP (MD will) (VP (VB have) (NP (DT a) (NN picnic)) (SBAR (IN if) (S (NP (PRP it)) (VP (VBZ is) (NP (DT a) (JJ sunny) (NN day)) (NP (NN tomorrow))))))) (. .)))",\
        "tmpl": "(ROOT (S (S ) (, ) (CC ) (S ) (. )))"}'


response, accept = output_fn(
    predict_fn(
        input_fn(payload, "application/json"),
        model_fn("../")
    ),
    "application/json"
)
json.loads(response).keys()
