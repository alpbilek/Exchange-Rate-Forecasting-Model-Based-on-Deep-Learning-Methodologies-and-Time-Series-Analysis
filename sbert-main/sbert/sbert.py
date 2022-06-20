# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import os
import logging
import json
import numpy as np
import torch
from torch import nn
from collections import OrderedDict
from . import __MODEL_HUB_ORGANIZATION__
from .utils.log import logger
from .utils.hug_model import snapshot_download
from .version import __versin__
from tqdm.autonotebook import trange
from .utils.util import batch_to_device, import_from_string
from .models.transformer import Transformer
from .models.polling import Pooling


class SBert(nn.Sequential):
    def __init__(self, model_name_or_path=None,
                 modules=None,
                 device=None,
                 cache_folder=None):
        self._model_config = {}

        if cache_folder is None:
            try:
                from torch.hub import _get_torch_home

                torch_cache_home = _get_torch_home()
            except ImportError:
                torch_cache_home = os.path.expanduser(
                    os.getenv('TORCH_HOME', os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')))
            cache_folder = os.path.join(torch_cache_home, 'sentence_transformers')

        if model_name_or_path is not None and model_name_or_path != "":
            logger.info('Load pretrained model:{}'.format(model_name_or_path))
            if os.path.exists(model_name_or_path):
                model_path = model_name_or_path
            else:
                if '\\' in model_name_or_path or model_name_or_path.count('/') > 1:
                    raise ValueError("Path {} error.".format(model_name_or_path))
                if '/' not in model_name_or_path:
                    model_name_or_path = __MODEL_HUB_ORGANIZATION__ + '/' + model_name_or_path
                model_path = os.path.join(cache_folder, model_name_or_path.replace('/', '_'))

                if not os.path.exists(model_path):
                    # Download model
                    model_path_t = snapshot_download(model_name_or_path,
                                                     cache_dir=cache_folder,
                                                     library_name=__MODEL_HUB_ORGANIZATION__,
                                                     library_version=__versin__,
                                                     ignore_files=['tf_model.h5', 'flax_model.msgpack',
                                                                   'rust_model.ot'])
                    os.rename(model_path_t, model_path)
            if os.path.exists(os.path.join(model_path, 'modules.json')):
                modules = self._load_sbert_model(model_path)
            else:
                modules = self._load_auto_model(model_path)

        if modules is not None and not isinstance(modules, OrderedDict):
            modules = OrderedDict([(str(idx), module) for idx, module in enumerate(modules)])

        super().__init__(modules)
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info('Use device: {}'.format(device))
        self._target_device = torch.device(device)

    def encode(self, sentences,
               batch_size=32,
               show_progress_bar=None,
               output_value='sentence_embedding',
               device=None,
               normalize_embeddings=False):
        """
        Get sentence embeddings

        :param sentences:
        :param batch_size:
        :param show_progress_bar:
        :param output_value:
        :param device:
        :param normalize_embeddings:
        :return:
        """
        self.eval()
        if show_progress_bar is None:
            show_progress_bar = logger.getEffectiveLevel() == logging.DEBUG

        if output_value == 'token_embeddings':
            pass

        input_is_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'):
            sentences = [sentences]
            input_is_string = True
        if device is None:
            device = self._target_device
        self.to(device)

        all_embeddings = []
        len_sorted_idx = np.argsort([-self._text_length(sent) for sent in sentences])  # big - small idx
        sentences_sorted = [sentences[idx] for idx in len_sorted_idx]

        for start_idx in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_idx:start_idx + batch_size]
            features = self.tokenize(sentences_batch)
            features = batch_to_device(features, device)
            with torch.no_grad():
                out_features = self.forward(features)
                if output_value == 'token_embeddings':
                    embeddings = []
                    for token_emb, attention in zip(out_features[output_value], out_features['attention_mask']):
                        last_mask_id = len(attention) - 1
                        while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                            last_mask_id -= 1
                        embeddings.append(token_emb[:last_mask_id + 1])
                else:
                    # Sentence embeddings
                    embeddings = out_features[output_value]
                    embeddings = embeddings.detach()
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                    embeddings = embeddings.cpu()

                all_embeddings.extend(embeddings)
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(len_sorted_idx)]
        # To numpy
        all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])
        if input_is_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def tokenize(self, texts):
        """
        Tokenizers texts
        :param texts:
        :return:
        """
        return self._first_module().tokenize(texts)

    def _text_length(self, text):
        """
        Get text len
        :param text:
        :return:
        """
        return sum([len(i) for i in text])

    def _load_auto_model(self, model_name_or_path):
        logger.warning("No sbert name found with name {}. Create MEAN pooling model".format(model_name_or_path))
        transformer_model = Transformer(model_name_or_path)
        pooling_model = Pooling(transformer_model)
        return [transformer_model, pooling_model]

    def _load_sbert_model(self, model_path):
        """
        Load sbert model
        :param model_path:
        :return:
        """
        config_json_path = os.path.join(model_path, 'config_sentence_transformers.json')
        if os.path.exists(config_json_path):
            with open(config_json_path, 'r', encoding='utf-8') as f:
                self._model_config = json.load(f)

            if '__version__' in self._model_config and 'sentence_transformers' in self._model_config['__version__'] \
                    and self._model_config['__version__']['sentence_transformers'] > __versin__:
                # logger.warning('Loaded model version {}, your version is {}'.format(
                #     self._model_config['__version__']['sentence_transformers'], __versin__))
                pass

        modules_json_path = os.path.join(model_path, 'modules.json')
        if os.path.exists(modules_json_path):
            with open(modules_json_path, 'r', encoding='utf-8') as f:
                modules_config = json.load(f)
        modules = OrderedDict()
        for m_config in modules_config:
            m_class = import_from_string(m_config['type'])
            m = m_class.load(os.path.join(model_path, m_config['path']))
            modules[m_config['name']] = m

        return modules

    def _first_module(self):
        """
        Get first module of sequential embedding
        :return:
        """
        return self._modules[next(iter(self._modules))]
