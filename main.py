from fastapi import FastAPI, Body, Request
import uvicorn
from doc2vec import initModel
from pydantic import BaseModel

class Document(BaseModel):
    id: str
    text: str 

app = FastAPI()

model, word2idx = initModel("model_state.pt")

@app.post("/analysis")
async def hello(document_list: list[Document]): 
    
    return len(document_list)

if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)