# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
from torch import nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
import json
from typing import List, Dict, Optional, Union, Tuple


class Transformer(nn.Sequential):
    """
    AutoModel to load. eg BERT
    """

    def __init__(self, model_name_or_path, max_seq_length,
                 model_args={}, cache_dir=None,
                 tokenizer_args={}, do_lower_case=False,
                 tokenizer_name_or_path=None):
        super(Transformer, self).__init__()
        self.config_keys = ['max_seq_length', 'do_lower_case']
        self.do_lower_case = do_lower_case

        config = AutoConfig.from_pretrained(model_name_or_path, **model_args, cache_dir=cache_dir)
        self.auto_model = AutoModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)
        if tokenizer_name_or_path is None:
            tokenizer_name_tmp = model_name_or_path
        else:
            tokenizer_name_tmp = tokenizer_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_tmp, cache_dir=cache_dir, **tokenizer_args)

        # Max seq len
        if max_seq_length is None:
            if hasattr(self.auto_model, 'config') \
                    and hasattr(self.auto_model.config, 'max_position_embeddings')\
                    and hasattr(self.tokenizer, 'model_max_length'):
                max_seq_length = min(self.auto_model.config.max_position_embeddings, self.tokenizer.model_max_length)
        self.max_seq_length = max_seq_length

        if tokenizer_name_or_path is not None:
            self.auto_model.config.tokenizer_class = self.tokenizer.__class__.__name__

    def forward(self, features):
        """
        Get token_embeddings, cls_token
        :param features:
        :return:
        """
        trans_features = {'input_ids': features['input_ids'], 'attention_mask': features['attention_mask']}
        if 'token_type_ids' in features:
            trans_features['token_type_ids'] = features['token_type_ids']

        out_states = self.auto_model(**trans_features, return_dict=False)
        out_tokens = out_states[0]

        cls_tokens = out_tokens[:, 0, :]  # CLS token is first token
        features.update({'token_embeddings': out_tokens, 'cls_token_embeddings': cls_tokens,
                         'attention_mask': features['attention_mask']})
        if self.auto_model.config.output_hidden_states:
            all_layer_idx = 2
            if len(out_states) < 3:
                all_layer_idx = 1
            hidden_states = out_states[all_layer_idx]
            features.update({'all_layer_embeddings': hidden_states})

        return features

    def get_word_embeddings_dimension(self):
        return self.auto_model.config.hidden_size

    def tokenize(self, texts):
        """
        Tokenizes text and maps tokens to token-ids
        :param texts:
        :return:
        """
        output = {}
        if isinstance(texts[0], str):
            to_tokenize = [texts]
        elif isinstance(texts[0], dict):
            to_tokenize = []
            output['text_keys'] = []
            for lookup in texts:
                text_key, text = next(iter(lookup.items()))
                to_tokenize.append(text)
                output['text_keys'].append(text_key)
            to_tokenize = [to_tokenize]
        else:
            batch1, batch2 = [], []
            for text_tupe in texts:
                batch1.append(text_tupe[0])
                batch2.append(text_tupe[1])
            to_tokenize = [batch1, batch2]

        # strip
        to_tokenize = [[str(s).strip() for s in col] for col in to_tokenize]
        if self.do_lower_case:
            to_tokenize = [[s.lower() for s in col] for col in to_tokenize]

        output.update(self.tokenizer(*to_tokenize, padding=True, truncation='longest_first',
                                     return_tensors='pt', max_length=self.max_seq_length))
        return output

    def get_config_dict(self):
        return {k: self.__dict__[k] for k in self.config_keys}

    def save(self, output_path):
        """
        Save model to output_path
        :param output_path:
        :return:
        """
        self.auto_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        with open(os.path.join(output_path, 'sentence_bert_config.json'), 'w', encoding='utf-8') as f:
            json.dump(self.get_config_dict(), f, indent=2)

    @staticmethod
    def load(input_path):
        """
        Load model from file path
        :param input_path:
        :return:
        """
        for config_name in ['sentence_bert_config.json', 'sentence_transformers_config.json']:
            sbert_config_path = os.path.join(input_path, config_name)
            if os.path.exists(sbert_config_path):
                break
        with open(sbert_config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return Transformer(model_name_or_path=input_path, **config)