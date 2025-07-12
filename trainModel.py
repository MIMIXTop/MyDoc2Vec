import torch
import time
from torch.utils.data import DataLoader
from doc2vec import Doc2VecDataset, Doc2Vec, collate_fn
from utils import del_stop_words

def train_model(texts: list[str], embed_dim: int, window: int, epochs: int, batch_size: int, lr: float):
    dataset = Doc2VecDataset(texts, window_size=window)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True
    )

    device: torch.device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")

    print(device.type) 

    model     = Doc2Vec(dataset.vocab_size, dataset.num_docs, embed_dim)
    model.to(device)
    loss_fn   = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    start = time.perf_counter()
    for epoch in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for doc_ids, word_ids, contexts in loader:
            doc_ids   = doc_ids.to(device, non_blocking=True)
            word_ids  = word_ids.to(device, non_blocking=True)
            contexts  = contexts.to(device, non_blocking=True)

            optimizer.zero_grad()
            log_probs = model(doc_ids, contexts)     
            loss      = loss_fn(log_probs, word_ids)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{epochs} — Loss: {avg_loss:.4f}")
    end = time.perf_counter()
    time_alg = (end - start) * 1_000_000
    print(f"Education time: {time_alg:.2f} microseconds" )
    return model, dataset