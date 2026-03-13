# 🎬 AI Movie Recommendation System

A powerful AI-driven movie recommendation system built with **FAISS vector search**, **Sentence Transformers**, and **Groq LLaMA 3.3** — featuring a clean Gradio web interface with three intelligent tabs.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange?logo=gradio)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-green)
![Groq](https://img.shields.io/badge/Groq-LLaMA%203.3-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📸 Features

| Tab | Feature |
|-----|---------|
| 🎯 **Recommendations** | Select genres + actor → get AI-curated movie picks |
| 💬 **Chat AI** | Conversational movie assistant powered by LLaMA 3.3 |
| 🔍 **Find Similar** | Enter a movie title → discover semantically similar films |

---

## 🛠️ Tech Stack

- **[Sentence Transformers](https://www.sbert.net/)** — `all-MiniLM-L6-v2` for semantic embeddings
- **[FAISS](https://faiss.ai/)** — Fast approximate nearest-neighbor vector search
- **[Groq API](https://groq.com/)** — Ultra-fast LLaMA 3.3 70B inference
- **[Gradio](https://www.gradio.app/)** — Interactive web UI
- **[Pandas](https://pandas.pydata.org/)** — Dataset handling (40,000 movies)

---

## 📁 Project Structure

```
ai-movie-recommender/
│
├── app.py                    # Main application
├── config.json               # API key config (not committed)
├── faiss_index.pkl           # Auto-generated vector index
├── requirements.txt          # Python dependencies
│
└── data/
    └── Movie_Dataset_40000.xlsx   # Movie dataset (40K entries)
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/ai-movie-recommender.git
cd ai-movie-recommender
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key

Create a `config.json` file in the root directory:

```json
{
  "GROQ_API_KEY": "your_groq_api_key_here"
}
```

> 🔑 Get your free API key at [console.groq.com](https://console.groq.com)

### 5. Add the dataset

Place your `Movie_Dataset_40000.xlsx` file inside the `data/` folder:

```
data/Movie_Dataset_40000.xlsx
```

### 6. Run the app

```bash
python app.py
```

The app will automatically build the FAISS index on first launch (this takes a few minutes), then open at `http://localhost:7860`.

---

## 📦 Requirements

Create a `requirements.txt` with:

```
gradio
pandas
numpy
faiss-cpu
sentence-transformers
groq
openpyxl
```

---

## 🔍 How It Works

### Hybrid Ranking Algorithm

When you search for movies, results are ranked using a weighted hybrid score:

```
score = 0.6 × semantic_similarity
      + 0.2 × normalized_rating
      + 0.1 × popularity_score
      + 0.1 × recency_score
```

This balances **relevance** (what matches your query) with **quality** (highly rated, popular, recent films).

### FAISS Vector Index

On first run, all 40,000 movie descriptions are encoded into 384-dimensional vectors using `all-MiniLM-L6-v2` and stored in a FAISS `IndexFlatL2` index. The index is cached as `faiss_index.pkl` for instant reloads.

---

## 💡 Usage Examples

**Recommendations Tab:**
- Select genres: `Action`, `Sci-Fi`
- Enter actor: `Tom Hanks`
- Click **Recommend Movies**

**Chat AI Tab:**
- *"Recommend me a horror movie with a psychological twist"*
- *"What are some good family-friendly animated films?"*
- *"I want something like Inception but more emotional"*

**Find Similar Movie Tab:**
- Enter: `The Dark Knight`
- Get 5 semantically similar films instantly

---

## 📊 Dataset

The system uses a curated dataset of **40,000 movies** with the following fields:

| Column | Description |
|--------|-------------|
| `movie_name` | Film title |
| `movie_year` | Release year |
| `director` | Director name |
| `rating_out_of_10` | IMDb-style rating |
| `movie_genre` | Genre category |
| `age_restriction` | G / PG / PG-13 / R / 18+ |
| `cast` | Lead actors |
| `overview_description` | Plot summary |
| `popularity_views` | View count |
| `tags` | Descriptive tags |

---

## 🚀 Deployment

To share publicly via Gradio:

```python
demo.launch(share=True)
```

Or deploy on **Hugging Face Spaces** by pushing this repo and setting your `GROQ_API_KEY` as a Space secret.

---

## 🔒 Security Note

Never commit your `config.json` to GitHub. Add it to `.gitignore`:

```
config.json
faiss_index.pkl
*.pkl
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you'd like to change.

1. Fork the repo
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Hugging Face](https://huggingface.co/) for Sentence Transformers
- [Meta AI](https://ai.meta.com/) for LLaMA 3.3
- [Groq](https://groq.com/) for blazing-fast inference
- [Facebook Research](https://github.com/facebookresearch/faiss) for FAISS

---

<p align="center">Made with ❤️ using Python & AI</p>
