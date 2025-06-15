from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np

# تحميل الموديل
# model = SentenceTransformer("local_models/multilingual-e5-small")
model = SentenceTransformer("intfloat/multilingual-e5-small")


app = FastAPI()

# الموديل بتاع الطلب
class TextInput(BaseModel):
    text: str

@app.post("/embed")
async def get_embedding(data: TextInput):
    embedding = model.encode(data.text)
    return {"embedding": embedding.tolist()}
