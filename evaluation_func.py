import numpy as np

def hit_rate(predicted_ranking, true_items,k):
    """
    Hit Rate（HR）

    parameters：
    - predicted_ranking: 
    - true_items: 

    return：
    - hr: Hit Rate
    """
    if k is not None:
        predicted_ranking = predicted_ranking[:k]
    hits = 0
    for item in predicted_ranking:
        if item in true_items:
            hits += 1
    hr = hits / len(true_items) if len(true_items) > 0 else 0
    return hr

def ndcg(predicted_ranking, true_items, k=None):
    """
    Normalized Discounted Cumulative Gain（NDCG）

    parameters：
    - predicted_ranking: 
    - true_items: 
    - k: 

    return：
    - ndcg_score: NDCG
    """
    if k is not None:
        predicted_ranking = predicted_ranking[:k]

    dcg = 0
    idcg = 0

    # dcg
    for i, item in enumerate(predicted_ranking):
        if item in true_items:
            dcg += 1 / np.log2(i + 2)
    
    # idcg
    true_ranking = np.zeros_like(predicted_ranking)
    for i, item in enumerate(predicted_ranking):
        true_ranking[i] = 1 if item in true_items else 0

    true_ranking=np.sort(true_ranking)[::-1]
    for i, item in enumerate(true_ranking):
        idcg += item / np.log2(i + 2)

    ndcg_score = dcg / idcg if idcg > 0 else 0
    return ndcg_score




def hr_ndcg(testdata, predictions, k=None):
    """

    parameters：
    - testdata: userid, itemid, rating（0/1）
    - predictions: userid, itemid, rating（0/1）
    - k:

    :return：
    - hr_score: Hit Rate
    - ndcg_score: NDCG
    """
    # predictions{userid: [itemid1, itemid2, ...], ...}

    sorted_predictions = predictions[np.argsort(predictions[:, 2],kind='mergesort')[::-1]]

    # dict
    recommendations = {}
    for user_id, item_id, _ in sorted_predictions:
        if user_id not in recommendations:
            recommendations[user_id] = []
        recommendations[user_id].append(item_id)
    hr_sum = 0
    ndcg_sum = 0
    total_users = len(set(testdata[:, 0]))

    for user_id in set(testdata[:, 0]):
        true_items = set(testdata[(testdata[:, 0] == user_id) & (testdata[:, 2] == 1)][:, 1])
        if user_id in recommendations:
            predicted_ranking = recommendations[user_id]
            hr_sum += hit_rate(predicted_ranking, true_items,k)
            ndcg_sum += ndcg(predicted_ranking, true_items, k)

    hr_score = hr_sum / total_users
    ndcg_score = ndcg_sum / total_users

    return hr_score, ndcg_score
