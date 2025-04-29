# 🎬 Video Recommendation System Backend

A simple backend for a movie recommendation system built with **FastAPI**. It provides movie recommendations to users using two techniques:

- **Collaborative Filtering**
- **Content-Based Filtering**

---

## 🚀 Features

- RESTful API using FastAPI
- Recommend movies based on user preferences and movie metadata
- Lightweight and easy to deploy
- Modular design for extensibility

---

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/satzurajkumar/a-simple-hybrid-movie-recommendation-engine.git
   cd a-simple-hybrid-movie-recommendation-engine


---
# 🛠️ Usage

1.**Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```
2.**Install dependencies:**
   ```bash
   pip install -r requirements.txt
```
3.**Run the FastAPI server:**
   ```bash
   uvicorn main:app --reload
```
4.**Access the API Docs:**

Swagger UI: http://localhost:8000/docs
ReDoc: http://localhost:8000/redoc
---
## 📁 Project Structure
```bash
.
├── main.py                # FastAPI application
├── models/                # Data models (Pydantic)
├── recommendation/        # Recommendation logic
│   ├── collaborative.py
│   └── content_based.py
├── data/                  # Sample data (users, movies, ratings)
├── requirements.txt       # Python dependencies
└── README.md
```


