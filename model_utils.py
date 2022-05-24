import torch
import torch.nn as nn
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import transformers
import warnings
warnings.filterwarnings("ignore") # Disable transformer pipeline warnings

import os
os.environ['TRULENS_BACKEND'] = 'torch'
from trulens.visualizations import NLP
from trulens.nn.models import get_model_wrapper
from trulens.nn.quantities import MaxClassQoI, ClassQoI, ComparativeQoI, LambdaQoI
from trulens.nn.attribution import IntegratedGradients, InputAttribution, Cut, OutputCut
from trulens.nn.distributions import PointDoi, GaussianDoi, LinearDoi
from trulens.utils.nlp import token_baseline

import shap

import numpy as np
import itertools

from IPython.display import display


'''
Utilities for loading and setting up models and tokenizer
'''

MODEL_FOLDER = "models"
MAX_TWEET_LEN = 128
LABELS = ["real", "fake"]
REAL = LABELS.index("real")
FAKE = LABELS.index("fake")


'''
Loads all relevant trained models and tokenizers, along with functions for using them
'''
class NLPHandler:
    def __init__(self, model_savename="BERT.ckpt"):
        self.classifier = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(DEVICE)
        
        if model_savename is not None and os.path.exists(f"{MODEL_FOLDER}/{model_savename}"):
            print(f"Loading trained classifier {model_savename}")
            self.classifier.load_state_dict(torch.load(f"{MODEL_FOLDER}/{model_savename}"))
        
        self.tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.tokenize = lambda tweets: self.tokenizer(tweets, add_special_tokens=True, padding="max_length", max_length=MAX_TWEET_LEN, truncation=True, return_tensors="pt", return_attention_mask=True).to(DEVICE)
        
        # Trulens model wrapper
        self.classifier_wrapper = get_model_wrapper(self.classifier, input_shape=(None, MAX_TWEET_LEN), device=DEVICE)
    
        # Trulens NLP visualizer
        self.visualizer = NLP(
            wrapper = self.classifier_wrapper,
            labels = LABELS,
            decode = lambda x: self.tokenizer.decode(x),
            tokenize = self.tokenize,
            input_accessor=lambda x: x['input_ids'],
            output_accessor=lambda x: x['logits'],
            hidden_tokens=set([self.tokenizer.pad_token_id])
        )
        
        
    '''
    Wrapper for computing the loss of the classifier on a batch
    of labeled tweets.
    Returns loss tensor on gpu.
    '''
    def loss(self, tweets, labels):
        labels = torch.LongTensor(labels).to(DEVICE)
        model_inputs = self.tokenize(tweets)
        model_outputs = self.classifier(**model_inputs, labels=labels)
        return model_outputs["loss"]
        
        
    '''
    Wrapper for running the classifier on batch of tweets.
    Returns detached np array of output logits
    '''
    def classify(self, tweets):
        if isinstance(tweets, np.ndarray):
            tweets = tweets.tolist()
            
        model_inputs = self.tokenize(tweets)
        model_outputs = self.classifier(**model_inputs)
        return model_outputs["logits"].detach().cpu().numpy()
    
    
    '''
    Return embeddings for particular words
    '''
    def get_embeddings(self, words):
        return self.classifier.get_input_embeddings()(self.tokenize(words)["input_ids"][:, 1]).detach().cpu().numpy()
        
        
        
    '''
    Create shap explainer object
    '''
    def shap_explainer(self):
        return shap.Explainer(
            self.classify,
            self.tokenizer,
            output_names=LABELS
        )
        
    '''
    Display attribution for a list of tweets
    '''
    def show_attributions(self, tweets):
        torch.cuda.empty_cache()
        display(self.visualizer.tokens(tweets, attributor=self.attributor()))
        torch.cuda.empty_cache()
    
    
    '''
    Create a type of input attributor
    '''
    def attributor(self, qoi=None):
        _, baseline_embedding_converter = token_baseline(
            keep_tokens=set([self.tokenizer.cls_token_id, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id]),
            replacement_token=self.tokenizer.pad_token_id,
            input_accessor=lambda x: x.kwargs['input_ids'],
            ids_to_embeddings=self.classifier.get_input_embeddings()
        )
        
        input_layer_name = self.classifier_wrapper._layernames[0]
        doi_cut = Cut(input_layer_name)
        
        doi = LinearDoi(
            baseline=baseline_embedding_converter,
            resolution=20,
            cut=doi_cut
        )
        
        if qoi is None:
            qoi = ComparativeQoI(FAKE, REAL)
        qoi_cut = OutputCut(accessor=lambda o: o['logits'])
        
        return InputAttribution(
            model=self.classifier_wrapper,
            doi=doi,
            doi_cut=doi_cut,
            qoi=qoi,
            qoi_cut=qoi_cut
        )
        
        
        
        
'''
Utility Function for concatenating shap explanation objects generated independently (assumed to be the same type)
'''
def explanation_concat(*explanations):
    def concat(field):
        val_list = [getattr(e, field, None) for e in explanations]
        
        if any(v is None for v in val_list):
            return None
        
        if isinstance(val_list[0], np.ndarray):
            return np.concatenate(val_list)
        
        if isinstance(val_list[0], list):
            return list(itertools.chain(*val_list))
        
        if isinstance(val_list[0], tuple):
            return tuple(itertools.chain(*val_list))
        
        raise ValueError(f"Unknown value type for field {field}: {type(val_list[0])}")
    
    def shared(field):
        val_list = [getattr(e, field, None) for e in explanations]
        
        if any(v is None for v in val_list):
            return None
        
        if all(v == val_list[0] for v in val_list):
            return val_list[0]
        
        raise ValueError(f"Shared field {field} has differences: {val_list}")
    
    return shap.Explanation(
        values=concat("values"),
        base_values=concat("base_values"),
        data=concat("data"),
        display_data=concat("display_data"),
        instance_names=concat("instance_names"),
        feature_names=concat("feature_names"),
        output_names=shared("output_names"),
        output_indexes=shared("output_indexes"),
        lower_bounds=None,
        upper_bounds=None,
        error_std=concat("error_std"),
        main_effects=concat("main_effects"),
        hierarchical_values=concat("hierarchical_values"),
        clustering=concat("clustering")
    )