from pipeline.loader import load_movies
from pipeline.embedder import to_documents
from pipeline.ingestor import build_vectorstore

movies = load_movies("data/raw/tmdb_5000_movies.csv")
docs   = to_documents(movies)
build_vectorstore(docs)
