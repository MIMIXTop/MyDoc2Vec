from returns.result import Result, Success, Failure
from trainModel import train_model
import utils


def main():
    FILENAME = "data.csv"
    
    model, ds = train_model(
        texts=utils.read_csv_texts(FILENAME, "text", limit=100),
        embed_dim= 50,
        window= 4,
        epochs= 15,
        batch_size= 64,
        lr= 0.01
        )
    
    device = next(model.parameters()).device

    documents_path = [
        "TEXT/doc1.txt",
        "TEXT/doc2.txt",
        "TEXT/doc3.txt",
        "TEXT/doc4.txt",
        "TEXT/doc5.txt",
        "TEXT/doc6.txt",
        "TEXT/doc7.txt",
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

if __name__ == "__main__":
    main()
