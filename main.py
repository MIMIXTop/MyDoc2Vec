from returns.result import Result, Success, Failure
from trainModel import train_model
import utils
import torch
import doc2vec

def main():
    FILENAME = "data.csv"
    embedding_dim = 250
    window = 6
    model, ds = train_model(
        texts=utils.read_csv_texts(FILENAME, "text", limit=500),
        embed_dim=embedding_dim,
        window=window,
        epochs=50,
        batch_size=512,
        lr=0.008
        )
    
    model.eval()
    device = next(model.parameters()).device

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
        doc_vec_list.append(utils.get_doc_vec(document=doc, word2idx=ds.word2idx, word_embeddings=model.word_embed, device=device))

    for i in range(len(doc_vec_list)):
        for j in range(len(doc_vec_list)):
            print(f"Cosine similarity {i + 1} with {j + 1}: {utils.cos_sim(doc_vec_list[i], doc_vec_list[j]):.2f}")
        print()

    print(f"Vocad size: {ds.vocab_size}")
    torch.save({
        "model_state_dict": model.state_dict(),
        "word2idx" : ds.word2idx,
        "vocab_size" : ds.vocab_size,
        "num_docs": ds.num_docs,
        "embedding_dim": embedding_dim,
        "window": window
    }, "model_state.pt")

    #model.to("cpu")
    #example_doc_ids = torch.tensor([0])  
    #example_context_ids = torch.tensor([[1, 2, 3, 4]])  
    
    # try:
    #     trace_model = torch.jit.trace(model, (example_doc_ids, example_context_ids))
    #     model_name = "doc2vec_model.pt"
    #     torch.jit.save(trace_model, model_name)
    #     print(f"Model saved in {model_name}")
    # except Exception as e:
    #     print(f"Error tracing model: {e}")
    #     return

    # with open("word2idx.txt", "w") as file:
    #     for word, idx in ds.word2idx.items():
    #         file.write(f"{word} {idx}\n")
    # print("Dataset saved in word2idx.txt")


if __name__ == "__main__":
    main()
