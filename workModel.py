import torch
import utils
from doc2vec import Doc2Vec, Doc2VecDataset
from returns.result import Success, Failure, Result



def main():
    checkpoint = torch.load("model_state.pt", map_location="cpu")

    model = Doc2Vec(
        vocab_size= checkpoint["vocab_size"],
        num_docs=checkpoint["num_docs"],
        embed_dim=checkpoint["embedding_dim"]        
    )  

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval() 

    word2idx = checkpoint["word2idx"]

    documents_path = [
        "TEXT/testText.txt",
        "TEXT/testText2.txt",
        "TEXT/testText3.txt",
        "TEXT/testText4.txt",
    ]

    documents_text = list[str]
    doc_res = [ text for path in documents_path for text in [utils.doc_reader(path)]] 
    documents_text = []

    for res in doc_res:
        match res:
            case Success(text):
                documents_text.append(text)
            case Failure(error):
                print(f"Failed: {error}")
    
    doc_vec_list = []

    for doc in documents_text:
        doc_vec_list.append(utils.get_doc_vec(document=doc, word2idx=word2idx, word_embeddings=model.word_embed, device=torch.device("cpu")))

    for i in range(len(doc_vec_list)):
        for j in range(len(doc_vec_list)):
            print(f"Cosine similarity {i + 1} with {j + 1}: {utils.cos_sim(doc_vec_list[i], doc_vec_list[j]):.2f}")
        print()



if __name__ == "__main__":
    main()
