from copy import deepcopy
from collate_fn import gen_eval_batch, gen_train_batch
from trainer import trainer
from evaluator import evaluator
from utils import reset_random_seed, LadderSampler
from torch.utils.data import DataLoader


def train_multi(model, max_len, n_epoch, train_data, eval_data, train_bsz, eval_bsz, optimizer, loss_fn, device, result_path, model_path):
    reset_random_seed(1)
    eval_loader = DataLoader(dataset=eval_data, batch_size=eval_bsz, num_workers=4,
                             collate_fn=lambda e: gen_eval_batch(e, eval_data, max_len))
    best_metric = evaluator(0, eval_loader, eval_data.n_item, model, device)
    i = 0
    for epoch in range(n_epoch):
        train_loader = DataLoader(dataset=train_data, batch_size=train_bsz,
                                  sampler=LadderSampler(train_data, train_bsz),
                                  num_workers=4,
                                  collate_fn=lambda e: gen_train_batch(e, train_data, max_len))
        optimizer = trainer(epoch, train_loader, model, optimizer, loss_fn, device)

        eval_loader = DataLoader(eval_data, batch_size=eval_bsz, num_workers=4,
                                 collate_fn=lambda e: gen_eval_batch(e, eval_data, max_len))
        current_metric = evaluator(eval_loader, eval_data.n_item, model, device)

        # if current_metric['HR@20'] >= best_metric['HR@20']:
        if current_metric['HR@10'] >= best_metric['HR@10']:
            i = 1
            best_metric = current_metric
            f = open(result_path, 'w')
            print('-' * 10, 'Best Metric', '-' * 10, file=f)
            print(best_metric, file=f)
            f.close()
            best_model = deepcopy(model)
            best_model.save(model_path)
        else:
            i = i + 1
            if i == 10:
                print('+' * 10, 'Early Stop', '+' * 10)
                break
    f = open(result_path, 'w')
    print(best_metric)
    print('-' * 10, 'Best Metric', '-' * 10, file=f)
    print(best_metric, file=f)
    f.close()