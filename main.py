from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import uvicorn
from doc2vec import initModel
from pydantic import BaseModel
from utils import document_analisys, serialize_to_json, MyDoc


class Document(BaseModel):
    id: str
    text: str 

app = FastAPI()

model, word2idx = initModel("model_state.pt")

@app.post("/analysis")
async def hello(document_list: list[Document]): 
    vec = [MyDoc(doc.id, doc.text) for doc in document_list]
    sim_vec = document_analisys(vec, word2idx=word2idx, word_embeddings=model.word_embed,  device="cpu")
    temp_data_for_json = serialize_to_json(sim_vec);
    return JSONResponse(content=jsonable_encoder(temp_data_for_json))

if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)