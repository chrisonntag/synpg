import os, argparse, codecs
import json
from urllib.parse import urlsplit
from synpg.utils import sent2str, synt2str, load_dictionary
import numpy as np
import torch
from nltk import ParentedTree
from synpg.subwordnmt.apply_bpe import BPE, read_vocabulary
from synpg.model import SynPG
from synpg.utils import Timer, make_path, load_data, load_embedding, load_dictionary, tree2tmpl, getleaf, synt2str, reverse_bpe
from tqdm import tqdm
from pprint import pprint


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

args = {
    'synpg_model_path': "./model/pretrained_synpg.pt",
    'pg_model_path': "./model/pretrained_parse_generator.pt",
    'bpe_codes_path': './data/bpe.codes',
    'bpe_vocab_path': './data/vocab.txt',
    'bpe_vocab_thresh': 50,
    'dictionary_path': "./data/dictionary.pkl",
    'max_sent_len': 40,
    'max_tmpl_len': 100,
    'max_synt_len': 160,
    'temp': 0.5,
    'seed': 97,
}

# fix random seed
np.random.seed(args['seed'])
torch.manual_seed(args['seed'])
torch.backends.cudnn.enabled = False


def template2tensor(templates, max_tmpl_len, dictionary):
    tmpls = np.zeros((len(templates), max_tmpl_len+2), dtype=np.long)
    for i, tp in enumerate(templates):
        tmpl_ = ParentedTree.fromstring(tp)
        tree2tmpl(tmpl_, 1, 2)
        tmpl_ = str(tmpl_).replace(")", " )").replace("(", "( ").split(" ")
        tmpl_ = [dictionary.word2idx[f"<{w}>"] for w in tmpl_ if f"<{w}>" in dictionary.word2idx]
        tmpl_ = [dictionary.word2idx["<sos>"]] + tmpl_ + [dictionary.word2idx["<eos>"]]
        tmpls[i, :len(tmpl_)] = tmpl_
    
    tmpls = torch.from_numpy(tmpls).to(device=torch.device(DEVICE))
    
    return tmpls
 

def model_fn(model_dir):
    """
    This function is the first to get executed upon a prediction request,
    it loads the model from the disk and returns the model object which will be used later for inference.
    """
    bpe_codes = codecs.open(model_dir+'/bpe.codes', encoding='utf-8')
    bpe_vocab = codecs.open(model_dir+'/vocab.txt', encoding='utf-8')
    bpe_vocab = read_vocabulary(bpe_vocab, args['bpe_vocab_thresh'])
    bpe = BPE(bpe_codes, '@@', bpe_vocab, None)

    # load dictionary and models
    dictionary = load_dictionary(model_dir+'/dictionary.pkl')

    synpg_model = SynPG(len(dictionary), 300, word_dropout=0.4)
    synpg_model.load_state_dict(torch.load(model_dir+'/pretrained_synpg.pt', map_location=torch.device(DEVICE)))
    synpg_model = synpg_model.to(device=torch.device(DEVICE))
    synpg_model.eval()

    pg_model = SynPG(len(dictionary), 300, word_dropout=0.4)
    pg_model.load_state_dict(torch.load(model_dir+'/pretrained_parse_generator.pt', map_location=torch.device(DEVICE)))
    pg_model = pg_model.to(device=torch.device(DEVICE))
    pg_model.eval()


    return synpg_model, pg_model, bpe, dictionary


def input_fn(request_body, request_content_type):
    """
    The request_body is passed in by SageMaker and the content type is passed in
    via an HTTP header by the client (or caller). This function then processes the 
    input data, and extracts three fields from the json body called "sent", "synt" 
    and "tmpl" and returns all three. 

    Example JSON input:
    {
        "sent": "The quick brown fox jumps over the lazy dog",
        "synt": "(ROOT (S (NP (DT The) (JJ quick) (JJ brown) (NN fox)) (VP (VBZ jumps) (PP (IN over) (NP (DT the) (JJ lazy) (NN dog)))))",
        "tmpl": "(ROOT (S (S ) (, ) (CC ) (S ) (. )))"
    }
    """
    # Check if content type is JSON
    if request_content_type == "application/json":
        request = json.loads(request_body)
    else:
        raise ValueError(f"Content type {request_content_type} is not supported")

    # Extract the sent, synt and tmpl from the request
    sent = request["sent"]
    synt = request["synt"]
    tmpl = request["tmpl"]


    return sent, synt, tmpl


def predict_fn(input_data, model):
    """
    This function takes in the input data and the model returned by the model_fn
    It gets executed after the model_fn and its output is returned as the API response.
    """

    synpg_model, pg_model, bpe, dictionary = model

    sent, synt, tmpl = input_data
    tmpls = template2tensor([tmpl], args['max_tmpl_len'], dictionary)

    with torch.no_grad():
        # convert syntax to tag sequence
        tagss = np.zeros((len(tmpls), args['max_sent_len']), dtype=np.long)
        tags_ = ParentedTree.fromstring(synt)
        tags_ = getleaf(tags_)
        tags_ = [dictionary.word2idx[f"<{w}>"] for w in tags_ if f"<{w}>" in dictionary.word2idx]
        tagss[:, :len(tags_)] = tags_[:args['max_sent_len']]
        
        tagss = torch.from_numpy(tagss).to(device=torch.device(DEVICE))
        
        # generate parses from tag sequence and templates
        parse_idxs = pg_model.generate(tagss, tmpls, args['max_synt_len'], temp=args['temp'])
        
        # add <sos> and remove tokens after <eos>
        synts = np.zeros((len(tmpls), args['max_synt_len']+2), dtype=np.long)
        synts[:, 0] = 1
        
        for i in range((len(tmpls))):
            parse_idx = parse_idxs[i].to(device=torch.device(DEVICE)).numpy()
            eos_pos = np.where(parse_idx==dictionary.word2idx["<eos>"])[0]
            eos_pos = eos_pos[0]+1 if len(eos_pos) > 0 else len(parse_idx)
            synts[i, 1:eos_pos+1] = parse_idx[:eos_pos]
            
        synts = torch.from_numpy(synts).to(device=torch.device(DEVICE))
        
        # bpe segment and convert sentence to tensor
        sents = np.zeros((len(tmpls), args['max_sent_len']), dtype=np.long)
        sent_ = bpe.segment(sent).split()
        sent_ = [dictionary.word2idx[w] if w in dictionary.word2idx else dictionary.word2idx["<unk>"] for w in sent_]
        sents[:, :len(sent_)] = sent_[:args['max_sent_len']]
        sents = torch.from_numpy(sents).to(device=torch.device(DEVICE))
        
        # generate paraphrases from sentence and generated parses
        output_idxs = synpg_model.generate(sents, synts, args['max_sent_len'], temp=args['temp'])
        output_idxs = output_idxs.to(device=torch.device(DEVICE)).numpy()
        
        
    return output_idxs, dictionary


def output_fn(prediction, accept):
    """
    Post-processing function for model predictions. It gets executed after the predict_fn 
    and returns the prediction as json.
    """

    # Check if accept type is JSON
    if accept != "application/json":
        raise ValueError(f"Accept type {accept} is not supported")

    output_idxs, dictionary = prediction
    paraphrase = reverse_bpe(synt2str(output_idxs[0], dictionary).split())

    return json.dumps(paraphrase), accept
