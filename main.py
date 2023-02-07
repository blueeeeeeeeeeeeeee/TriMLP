from evaluator import evaluator
from inter_data import InterData, un_serialize
from triangular_mixer import TriangularMixer
from seq_recommender import Recommender
from collate_fn import gen_eval_batch
from torch.utils.data import DataLoader


if __name__ == '__main__':
    data_path = 'Data/Amazon/Beauty/Beauty.data'
    dataset = un_serialize(data_path)
    n_item = dataset.n_item
    seq_len = 64
    train_data, eval_data = dataset.partition(seq_len)

    n_session = 4
    model = Recommender(n_item, d_model=64, encoder=TriangularMixer(seq_len, n_session))
    model.load('Models/Amazon/Beauty/Beauty.pkl')
    device = 'cuda:0'
    model.to(device)
    eval_loader = DataLoader(dataset=eval_data, batch_size=128, num_workers=4, collate_fn=lambda e: gen_eval_batch(e, eval_data, seq_len))
    current_metric = evaluator(eval_loader, eval_data.n_item, model, device)