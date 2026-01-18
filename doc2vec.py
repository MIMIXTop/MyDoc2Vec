import torch
from torch import nn
from torch.utils.data import Dataset
from collections import Counter
from utils import del_stop_words , get_lemma

class Doc2VecDataset(Dataset):
    def __init__(self, texts, window_size=2):
        self.window = window_size
        self.data = []
        self.vocab = Counter()
        

        for doc_id, raw in enumerate(texts):
            tokens = get_lemma(raw)
            tokens = del_stop_words(tokens)
            
            if not tokens:
                continue

            self.vocab.update(tokens)

            for idx, word in enumerate(tokens):
                start  = max(0, idx - window_size)
                left   = tokens[start:idx]
                right  = tokens[idx+1: idx+1+window_size]
                context = left + right
                if not context:
                    continue
                self.data.append((doc_id, word, context))
        
        if not self.data:
            raise ValueError("No valid training examples found after preprocessing")

        self.word2idx = {w:i for i, w in enumerate(self.vocab)}
        self.idx2word = {i:w for w, i in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        self.num_docs   = len(texts)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        doc_id, word, context = self.data[idx]
        word_id     = self.word2idx[word]
        context_ids = [self.word2idx[t] for t in context]
        return doc_id, word_id, context_ids


class Doc2Vec(nn.Module):
    def __init__(self, vocab_size, num_docs, embed_dim):
        super().__init__()
        self.word_embed = nn.Embedding(vocab_size, embed_dim)
        self.doc_embed  = nn.Embedding(num_docs, embed_dim)
        self.fc         = nn.Linear(embed_dim, vocab_size)
        self.log_softmax= nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, doc_ids: torch.Tensor, context_ids: torch.Tensor):

        d_vec = self.doc_embed(doc_ids)                             
        c_vec = self.word_embed(context_ids).mean(dim=1)            
        h     = d_vec + c_vec  
        h = self.dropout(h)                                      
        out   = self.fc(h)
        return self.log_softmax(out)                              


def collate_fn(batch):
    docs, words, contexts = zip(*batch)
    docs_t  = torch.tensor(docs, dtype=torch.long)
    words_t = torch.tensor(words, dtype=torch.long)
    ctx_ts  = [torch.tensor(c, dtype=torch.long) for c in contexts]
    ctx_p   = nn.utils.rnn.pad_sequence(ctx_ts, batch_first=True, padding_value=0)
    return  docs_t, words_t, ctx_p


