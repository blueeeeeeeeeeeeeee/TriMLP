import time
import torch
import numpy as np
from collections import Counter
from tqdm import tqdm


def count(logits, trg, cnt):
    output = logits.clone()
    for i in range(trg.size(0)):
        output[i][0] = logits[i][trg[i]]
        output[i][trg[i]] = logits[i][0]
    idx = output.sort(descending=True, dim=-1)[1]
    order = idx.topk(k=1, dim=-1, largest=False)[1]
    cnt.update(order.squeeze().tolist())
    return cnt


def calculate(cnt, array):
    for k, v in cnt.items():
        array[k] = v
    hr = array.cumsum()
    ndcg = 1 / np.log2(np.arange(0, len(array)) + 2)
    ndcg = ndcg * array
    ndcg = ndcg.cumsum() / hr.max()
    hr = hr / hr.max()
    return hr, ndcg


def evaluator(epoch, data_loader, n_item, model, device):
    start_time = time.time()
    cnt = Counter()
    array = np.zeros(n_item)
    model.eval()
    with torch.no_grad():
        batch_iterator = tqdm(enumerate(data_loader), total=len(data_loader), leave=True)
        for _, (src_los, trg_locs, data_size) in batch_iterator:
            src = src_los.to(device)
            trg = trg_locs.to(device)
            dsz = data_size.to(device)
            logits = model(src, dsz)
            cnt = count(logits, trg, cnt)
    cost_time = time.time() - start_time
    hr, ndcg = calculate(cnt, array)
    print('Time={:.4f}, HR@1={:.4f}, HR@5={:.4f}, NDCG@5={:.4f}, HR@10={:.4f}, NDCG@10={:.4f}'
          .format(cost_time, hr[0], hr[4], ndcg[4], hr[9], ndcg[9]))
    metrics = {'Cost Time': '{:.4f}'.format(cost_time),
               'HR@5': '{:.5f}'.format(hr[4]), 'NDCG@5': '{:.5f}'.format(ndcg[4]),
               'HR@10': '{:.5f}'.format(hr[9]), 'NDCG@10': '{:.5f}'.format(ndcg[9])}
    # print('Time={:.4f}, HR@20={:.4f}, NDCG@20={:.4f}'.format(cost_time, hr[19], ndcg[19]))
    # metrics = {'Cost Time': cost_time, 'HR@20': hr[19], 'NDCG@20': ndcg[19]}
    return metrics