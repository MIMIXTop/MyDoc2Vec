import torch
import re, csv, time
from returns.result import Result, Success, Failure 
from natasha import Doc, MorphVocab, Segmenter, NewsEmbedding, NewsMorphTagger, NewsNERTagger, NamesExtractor, NewsSyntaxParser 

segmenter = Segmenter()
morph_vocab = MorphVocab()

emb = NewsEmbedding()
morph_tagger = NewsMorphTagger(emb)
syntax_parser = NewsSyntaxParser(emb)
ner_tagger = NewsNERTagger(emb)
name_extractor = NamesExtractor(morph_vocab)


def load_stop_words():
    with open("russian_stopwords.txt") as file:
        return {sp_word for sp_word in file}

STOP_WORDS = load_stop_words()

def del_stop_words(words: list[str]) -> list[str]:
    return [ token for token in words if token not in STOP_WORDS ]

def tokenizer(text: str) -> list[str]:
    return re.sub(r'[^а-я@#]', ' ', text.lower()).split()

def get_doc_vec(document: str, word2idx: dict[str,int], word_embeddings: torch.nn.Embedding, device: torch.device) -> torch.Tensor:
    tokens = get_lemma(document)
    tokens = del_stop_words(tokens)
    embed_list = []

    for word in tokens:
        if word in word2idx:
            idx = word2idx[word]
            input_data = torch.tensor(idx, dtype=torch.long, device=device) 
            embed = word_embeddings(input_data);
            embed_list.append(embed.squeeze(0))

    if not embed_list:
        return torch.zeros(word_embeddings.embedding_dim, device=device)
    
    stacked = torch.stack(embed_list)
    docVec = torch.mean(stacked, 0)
    return docVec

def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1).item()

def read_csv_texts(path: str, column: str, limit: int ) -> list[str]:
    texts = []
    with open(path, newline='', encoding='utf-8') as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= limit:
                break
            texts.append(row[column])
    return texts

def doc_reader(path: str) -> Result[str, str]:
    try:    
        with open(path, "r") as file:
            return Success(file.read()) 
    except:
        return Failure("Error file read")
    

def get_lemma(text: str) -> list[str] :
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.parse_syntax(syntax_parser)

    for token in doc.tokens:
        token.lemmatize(morph_vocab)
    
    return [token.lemma for token in doc.tokens if bool(re.fullmatch(r'^[a-zA-Zа-яА-ЯёЁ]+$', token.lemma))]

def document_analisys(documents: list[str]):
    
    
    
    return ...