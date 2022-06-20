# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import importlib
import queue

import numpy as np
import torch
from torch import Tensor, device


def batch_to_device(batch, target_device: device):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch


def import_from_string(string_path):
    """
    Import a dotted module path
    :param string_path: a.b
    :return: a/b
    """
    try:
        module_path, class_name = string_path.rsplit('.', 1)
    except ValueError:
        raise ImportError("error module path: %s" % string_path)

    try:
        module = importlib.import_module(string_path)
    except:
        module = importlib.import_module(module_path)

    try:
        return getattr(module, class_name)
    except AttributeError:
        raise ImportError("Module {} has no {} attribute/classs".format(module_path, class_name))


def cos_sim(a, b):
    """
    Cosine Similarity
    :param a: Tensor
    :param b: Tensor
    :return: res[i][j] = cos_sim(a[i], b[j])
    """
    if not isinstance(a, Tensor):
        a = torch.tensor(a)
    if not isinstance(b, Tensor):
        b = torch.tensor(b)
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


def dot_sim(a, b):
    """
    Dot-product Similarity
    :param a: Tensor
    :param b: Tensor
    :return: res[i][j] = dot_product(a[i], b[j])
    """
    if not isinstance(a, Tensor):
        a = torch.tensor(a)
    if not isinstance(b, Tensor):
        b = torch.tensor(b)
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    return torch.mm(a, b.transpose(0, 1))


def normalize_embeddings(embeddings):
    """
    Norm embedding matrix
    :param embeddings: Tensor
    :return:
    """
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


def semantic_search(query_embeddings,
                    corpus_embeddings,
                    query_chunk_size=100,
                    corpus_chunk_size=500000,
                    top_k=10,
                    score_function=cos_sim):
    """
    Cosine similarity search between a list of query embeddings  and a list of corpus embeddings.
    It can be used for Information Retrieval / Semantic Search for corpora up to about 1 Million entries.
    :param query_embeddings: A 2 dimensional tensor with the query embeddings.
    :param corpus_embeddings: A 2 dimensional tensor with the corpus embeddings.
    :param query_chunk_size: Process 100 queries simultaneously. Increasing that value increases the speed, but requires more memory.
    :param corpus_chunk_size: Scans the corpus 100k entries at a time. Increasing that value increases the speed, but requires more memory.
    :param top_k: Retrieve top k matching entries.
    :param score_function: Funtion for computing scores. By default, cosine similarity.
    :return: Returns a sorted list with decreasing cosine similarity scores. Entries are dictionaries with the keys 'corpus_id' and 'score'
    """
    if isinstance(query_embeddings, (np.ndarray, np.generic)):
        query_embeddings = torch.from_numpy(query_embeddings)
    elif isinstance(query_embeddings, list):
        query_embeddings = torch.stack(query_embeddings)
    if len(query_embeddings.shape) == 1:
        query_embeddings = query_embeddings.unsqueeze(0)

    if isinstance(corpus_embeddings, (np.ndarray, np.generic)):
        corpus_embeddings = torch.from_numpy(corpus_embeddings)
    elif isinstance(corpus_embeddings, list):
        corpus_embeddings = torch.stack(corpus_embeddings)

    # queries and corpus on the same device
    if corpus_embeddings.device != query_embeddings.device:
        query_embeddings = query_embeddings.to(corpus_embeddings.device)

    queries_results = [[] for _ in range(len(query_embeddings))]
    for query_start_idx in range(0, len(query_embeddings), query_chunk_size):
        for corpus_start_idx in range(0, len(corpus_embeddings), corpus_chunk_size):
            scores = score_function(query_embeddings[query_start_idx:query_start_idx + query_chunk_size],
                                    corpus_embeddings[corpus_start_idx:corpus_start_idx + corpus_chunk_size])
            # Get top-k scores
            scores_topk_values, score_topk_idx = torch.topk(scores, min(top_k, len(scores[0])),
                                                            dim=1, largest=True, sorted=False)
            scores_topk_values = scores_topk_values.cpu().tolist()
            scores_topk_idx = score_topk_idx.cpu().tolist()

            for query_idx in range(len(scores)):
                for sub_corpus_id, score in zip(scores_topk_idx[query_idx], scores_topk_values[query_idx]):
                    corpus_id = corpus_start_idx + sub_corpus_id
                    query_id = query_start_idx + query_idx
                    queries_results[query_id].append({'corpus_id': corpus_id, 'score': score})

    # Sort top_k results
    for idx in range(len(queries_results)):
        queries_results[idx] = sorted(queries_results[idx], key=lambda x: x['score'], reverse=True)
        queries_results[idx] = queries_results[idx][0:top_k]

    return queries_results


def community_detection(embeddings, threshold=0.75, min_community_size=10, init_max_size=1000):
    """
    Fast Community Detection

    Finds in the embeddings all communities, i.e. embeddings that are close (closer than threshold).
    Returns only communities that are larger than min_community_size. The communities are returned
    in decreasing order. The first element in each list is the central point in the community.
    :param embeddings:
    :param threshold:
    :param min_community_size:
    :param init_max_size:
    :return:
    """
    cos_scores = cos_sim(embeddings, embeddings)

    top_k_values, _ = cos_scores.topk(k=min_community_size, largest=True)
    # Filter for rows >= min_threshold
    extracted_communities = []
    for i in range(len(top_k_values)):
        if top_k_values[i][-1] >= threshold:
            new_cluster = []
            # topk most similar entity
            top_val_large, top_idx_large = cos_scores[i].topk(k=init_max_size, largest=True)
            top_idx_large = top_idx_large.tolist()
            top_val_large = top_val_large.tolist()

            if top_val_large[-1] < threshold:
                for idx, val in zip(top_idx_large, top_val_large):
                    if val < threshold:
                        break
                    new_cluster.append(idx)
            else:
                for idx, val in enumerate(cos_scores[i].tolist()):
                    if val > threshold:
                        new_cluster.append(idx)
            extracted_communities.append(new_cluster)
    # Largest cluster first
    extracted_communities = sorted(extracted_communities, key=lambda x: len(x), reverse=True)

    # Remove overlapping communities
    unique_communities = []
    extracted_ids = set()
    for community in extracted_communities:
        add_cluster = True
        for idx in community:
            if idx in extracted_ids:
                add_cluster = False
                break
        if add_cluster:
            unique_communities.append(community)
            for idx in community:
                extracted_ids.add(idx)

    return unique_communities


def paraphrase_mining(embeddings,
                      query_chunk_size=5000,
                      corpus_chunk_size=100000,
                      max_pairs=500000,
                      top_k=100,
                      score_function=cos_sim):
    """
    Given a list of sentences / texts, this function performs paraphrase mining. It compares all sentences against all
    other sentences and returns a list with the pairs that have the highest cosine similarity score.

    :param embeddings:
    :param query_chunk_size:
    :param corpus_chunk_size:
    :param max_pairs:
    :param top_k:
    :param score_function:
    :return: Returns a list of triplets with the format [score, id1, id2]
    """
    top_k += 1  # A sentence has the highest similarity to itself. Increase +1 as we are interest in distinct pairs

    # Mine for duplicates
    pairs = queue.PriorityQueue()
    min_score = -1
    num_added = 0

    for corpus_start_idx in range(0, len(embeddings), corpus_chunk_size):
        for query_start_idx in range(0, len(embeddings), query_chunk_size):
            scores = score_function(embeddings[query_start_idx: query_start_idx + query_chunk_size],
                                    embeddings[corpus_start_idx: corpus_start_idx + corpus_chunk_size])
            # Get top-k scores
            scores_topk_values, score_topk_idx = torch.topk(scores, min(top_k, len(scores[0])),
                                                            dim=1, largest=True, sorted=False)
            scores_topk_values = scores_topk_values.cpu().tolist()
            scores_topk_idx = score_topk_idx.cpu().tolist()
            for query_idx in range(len(scores)):
                for topk_idx, corpus_idx in enumerate(scores_topk_idx[query_idx]):
                    i = query_start_idx + query_idx
                    j = corpus_start_idx + corpus_idx

                    if i != j and scores_topk_values[query_idx][topk_idx] > min_score:
                        pairs.put((scores_topk_values[query_idx][topk_idx], i, j))
                        num_added += 1
                        if num_added >= max_pairs:
                            entry = pairs.get()
                            min_score = entry[0]

    # Get the pairs
    added_pairs = set()  # Used for duplicate detection
    pairs_list = []
    while not pairs.empty():
        score, i, j = pairs.get()
        sorted_i, sorted_j = sorted([i, j])
        if sorted_i != sorted_j and (sorted_i, sorted_j) not in added_pairs:
            added_pairs.add((sorted_i, sorted_j))
            pairs_list.append([score, i, j])

    # Sort with highest scores first
    pairs_list = sorted(pairs_list, key=lambda x: x[0], reverse=True)
    return pairs_list
