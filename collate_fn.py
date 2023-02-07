import torch
from utils import pad_sequence


def gen_train_batch(batch, data_source, max_len):
    src_seq, trg_seq = zip(*batch)
    locs, data_size = [], []
    for e in src_seq:
        _, l_, _ = zip(*e)
        locs.append(pad_sequence(l_, max_len))
        data_size.append(len(_))
    src_locs = torch.stack(locs)
    data_size = torch.tensor(data_size)
    locs = []
    for e in trg_seq:
        _, l_, _ = zip(*e)
        locs.append(pad_sequence(l_, max_len))
    trg_locs = torch.stack(locs)
    return src_locs, trg_locs, data_size


def gen_eval_batch(batch, data_source, max_len):
    src_seq, trg_seq = zip(*batch)
    locs, data_size = [], []
    for e in src_seq:
        _, l_, _ = zip(*e)
        locs.append(pad_sequence(l_, max_len))
        data_size.append(len(_))
    src_locs = torch.stack(locs)
    data_size = torch.tensor(data_size)
    locs = []
    for e in trg_seq:
        _, l_, _ = zip(*e)
        locs.append(pad_sequence(l_, 1))
    trg_locs = torch.stack(locs)
    return src_locs, trg_locs, data_size
