from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI()

embedder = SentenceTransformer('model-placeholder') #choose whichever model you seem fit.
data = pd.read_csv('dataset-placeholder.csv')
data = data.dropna(subset=['column_A'])
data = data[data['column_A'].str.strip() != '']
data['vector'] = data['column_A'].apply(lambda x: embedder.encode(x))

class Input(BaseModel):
    text: str

@app.post("/ask-query")
async def process(input_text: Input):
    user_vec = embedder.encode(input_text.text)
    vecs = np.vstack(data['vector'].to_numpy())
    sim_scores = cosine_similarity([user_vec], vecs)[0]
    match_idx = np.argmax(sim_scores)
    match_data = data.iloc[match_idx]
    response_data = match_data['column_B']
    info_data = match_data['column_C']
    response = {
        "reply": response_data,
    }
    if pd.notna(info_data) and info_data.strip() != '':
        response['reply'] += f"\nDetails: {info_data}"
    return response
