import time
from tqdm import tqdm


def trainer(epoch, data_loader, model, optimizer, loss_fn, device):
    print('+'*30, 'Epoch {}'.format(epoch+1), '+'*30)
    start_time = time.time()
    model.train()
    running_loss = 0.
    processed_batch = 0
    batch_iterator = tqdm(enumerate(data_loader), total=len(data_loader), leave=True)
    for batch_idx, (src_locs, trg_locs, data_size) in batch_iterator:
        optimizer.zero_grad()
        src = src_locs.to(device)
        trg = trg_locs.to(device)
        dsz = data_size.to(device)
        logits = model(src, dsz)
        logits = logits.view(-1, logits.size(-1))
        trg = trg.view(-1)
        loss = loss_fn(logits, trg)
        loss.backward()
        optimizer.step()
        running_loss += loss.detach().cpu().item()
        processed_batch = processed_batch + 1
        batch_iterator.set_postfix_str('Loss={:.4f}'.format(loss.item()))
    cost_time = time.time() - start_time
    avg_loss = running_loss / processed_batch
    print('Time={:.4f}, Average Loss={:.4f}'.format(cost_time, avg_loss))
    return optimizer