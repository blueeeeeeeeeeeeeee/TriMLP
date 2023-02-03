import torch
import torch.nn as nn
from Data.inter_data import InterData
from Data.inter_data import un_serialize
from TrainTools.train import train_multi
from abstract_model import Recommender
from Encoders import Square, Eye, Tri, MHTri, MSTri, Plus, MSTriOL, Serial, SerialR, Merge, PlusOL, SerialOL, SerialROL, MergeOL


if __name__ == '__main__':

    data_vocab = {0: ['MovieLens', '100K', '1M', '10M', '20M'],
                  1: ['Amazon', 'Beauty', 'Toys', 'Sports'],
                  2: ['Tenrec', 'QB_Article', 'QB_Video']}
    all_segments = [1, 64]
# **********************************************************************************************************************
    for i in range(3):
        data_family = data_vocab[i][0]
        for j in range(1, len(data_vocab[i]), 1):
            data_name = data_vocab[i][j]
# **********************************************************************************************************************
            prefix = '/home/jyh/TriMLP4Rec/Data/'
            data_path = prefix + data_family + '/' + data_name + '/' + data_name + '.data'
            dataset = un_serialize(data_path)
            n_item = dataset.n_item
# **********************************************************************************************************************
            for n_s in range(2):
                n_segment = all_segments[n_s]
                model_vocab = {
                    0: ['MSTri', MSTri(seq_len=64, n_segment=n_segment)],
                    1: ['PLus', Plus(seq_len=64, n_segment=n_segment)],
                    2: ['MSTri(OL)', MSTriOL(seq_len=64, n_segment=n_segment)],
                    3: ['Serial', Serial(seq_len=64, n_segment=n_segment)],
                    4: ['Serial(Reversed)', SerialR(seq_len=64, n_segment=n_segment)],
                    5: ['Merge', Merge(seq_len=64, n_segment=n_segment, d_model=64)],
                    6: ['Plus(OL)', PlusOL(seq_len=64, n_segment=n_segment)],
                    7: ['Serial(OL)', SerialOL(seq_len=64, n_segment=n_segment)],
                    8: ['SerialR(OL)', SerialROL(seq_len=64, n_segment=n_segment)],
                    9: ['Merge(OL)', MergeOL(seq_len=64, n_segment=n_segment, d_model=64)]}
                for k in range(10):
                    model_name = model_vocab[k][0] + '_' + str(n_segment)
                    model = Recommender(n_item, d_model=64, encoder=model_vocab[k][1])
# **********************************************************************************************************************
                    device = 'cuda:0'
                    model.to(device)
                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
                    n_epoch = 500
                    train_data, eval_data = dataset.partition(64)
                    train_bsz = 128
                    eval_bsz = 64
                    result_path = '/home/jyh/TriMLP4Rec/Results/' + data_family + '/' + data_name + '/' + model_name + '.txt'
                    model_path = '/home/jyh/TriMLP4Rec/Models/' + data_family + '/' + data_name + '/' + model_name + '.pkl'
                    train_multi(model, 64, n_epoch, train_data, eval_data, train_bsz, eval_bsz, optimizer, loss_fn, device,
                                result_path, model_path)
