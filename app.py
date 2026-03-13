import os
import json
import pickle
import pandas as pd
import numpy as np
import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer
from groq import Groq

# =========================
# PATH SETUP
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(BASE_DIR, "data", "Movie_Dataset_40000.xlsx")
INDEX_FILE = os.path.join(BASE_DIR, "faiss_index.pkl")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")

# =========================
# LOAD API KEY
# =========================

def load_api():
    with open(CONFIG_FILE) as f:
        return json.load(f)["GROQ_API_KEY"]

client = Groq(api_key=load_api())

# =========================
# LOAD EMBEDDING MODEL
# =========================

print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

STORE = None

# =========================
# LOAD DATASET
# =========================

def load_movies():

    df = pd.read_excel(DATA_FILE)

    df.columns = df.columns.str.lower()

    rename_map = {
        "movie_name":"title",
        "movie_year":"year",
        "rating_out_of_10":"rating",
        "movie_genre":"genre",
        "overview_description":"overview",
        "cast":"cast",
        "popularity_views":"popularity",
        "tags":"tags"
    }

    df = df.rename(columns=rename_map)

    for col in ["title","year","rating","genre","overview","cast","popularity","tags"]:
        if col not in df.columns:
            df[col] = ""

    df.fillna("", inplace=True)

    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0).astype(float)
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0).astype(int)

    df["text"] = (
        df["title"].astype(str) + " " +
        df["genre"].astype(str) + " " +
        df["overview"].astype(str) + " " +
        df["cast"].astype(str) + " " +
        df["tags"].astype(str)
    )

    print("Loaded", len(df), "movies")

    return df.reset_index(drop=True)

# =========================
# BUILD FAISS INDEX
# =========================

def build_index():

    df = load_movies()

    print("Encoding embeddings...")

    vectors = embedder.encode(df["text"].tolist(), convert_to_numpy=True)

    dim = vectors.shape[1]

    index = faiss.IndexFlatL2(dim)

    index.add(vectors)

    with open(INDEX_FILE,"wb") as f:
        pickle.dump({"index":index,"data":df},f)

    print("Index built:", index.ntotal)

# =========================
# LOAD INDEX
# =========================

def load_store():

    global STORE

    if STORE is None:

        if not os.path.exists(INDEX_FILE):
            build_index()

        with open(INDEX_FILE,"rb") as f:
            STORE = pickle.load(f)

    return STORE

# =========================
# HYBRID RANKING
# =========================

def hybrid_score(row):

    rating = row["rating"]/10

    pop = min(row["popularity"]/1000000,1)

    recency = max(0,(row["year"]-1980)/50)

    semantic = 1/(1+row["_score"])

    return (
        0.6*semantic +
        0.2*rating +
        0.1*pop +
        0.1*recency
    )

# =========================
# SEARCH MOVIES
# =========================

def search_movies(query,k=20):

    store = load_store()

    index = store["index"]
    df = store["data"]

    q = embedder.encode([query], convert_to_numpy=True)

    D,I = index.search(q,k)

    res = df.iloc[I[0]].copy()

    res["_score"] = D[0]

    res["hybrid_score"] = res.apply(hybrid_score,axis=1)

    res = res.sort_values("hybrid_score",ascending=False)

    return res

# =========================
# RECOMMENDATION
# =========================

def recommend(genres,actor):

    query = " ".join(genres) + " " + actor

    res = search_movies(query,30)

    ctx=""

    for _,r in res.head(10).iterrows():
        ctx += f"{r.title} ({r.year}) {r.genre} rating {r.rating}\n"

    prompt=f"""
Recommend 5 movies.

User likes genres: {genres}
Actor preference: {actor}

Movies:
{ctx}

Format:
Movie Title (Year)
Reason
"""

    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role":"user","content":prompt}],
        max_tokens=400
    )

    return resp.choices[0].message.content

# =========================
# CHAT RECOMMENDER
# =========================

def chat_recommend(message, history):

    movie_keywords = [
        "movie","film","recommend","watch",
        "actor","actress","genre","cinema",
        "hollywood","bollywood"
    ]

    is_movie_query = any(word in message.lower() for word in movie_keywords)

    if is_movie_query:

        res = search_movies(message, 10)

        ctx = ""
        for _, r in res.iterrows():
            ctx += f"{r.title} ({r.year}) {r.genre}\n"

        prompt = f"""
User query: {message}

Relevant movies:
{ctx}

Recommend 3 movies briefly and explain why they match the user request.
"""

    else:

        prompt = f"""
You are a friendly AI assistant.

User message:
{message}

Reply naturally and helpfully.
"""

    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300
    )

    answer = resp.choices[0].message.content

    if history is None:
        history = []

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})

    return "", history

# =========================
# SIMILAR MOVIES
# =========================

def similar(title):

    store = load_store()
    df = store["data"]

    row = df[df["title"].str.lower()==title.lower()]

    if row.empty:
        return "Movie not found"

    text = row.iloc[0]["text"]

    q = embedder.encode([text],convert_to_numpy=True)

    D,I = store["index"].search(q,6)

    out="Similar movies:\n"

    for idx in I[0][1:]:
        r=df.iloc[idx]
        out+=f"{r.title} ({r.year}) {r.genre}\n"

    return out

# =========================
# UI
# =========================

GENRES=[
"Action","Drama","Comedy","Sci-Fi","Horror",
"Romance","Thriller","Adventure","Fantasy",
"Mystery","Crime","Animation"
]

with gr.Blocks(title="AI Movie Recommendation System") as demo:

    gr.Markdown("# 🎬 AI Movie Recommendation System")

    with gr.Tab("Recommendations"):

        genres=gr.CheckboxGroup(GENRES,label="Select Genres")

        actor=gr.Textbox(label="Favorite Actor")

        btn=gr.Button("Recommend Movies")

        out=gr.Textbox(lines=15)

        btn.click(recommend,[genres,actor],out)

    with gr.Tab("Chat AI"):

        chatbot=gr.Chatbot(height=400)

        msg=gr.Textbox()

        msg.submit(chat_recommend,[msg,chatbot],[msg,chatbot])

    with gr.Tab("Find Similar Movie"):

        title=gr.Textbox(label="Movie Title")

        btn2=gr.Button("Find Similar")

        out2=gr.Textbox(lines=10)

        btn2.click(similar,title,out2)

demo.launch()