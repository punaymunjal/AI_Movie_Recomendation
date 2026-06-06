import pandas as pd
import json

def load_movies(path: str) -> list[dict]:
    df = pd.read_csv(path)

    def parse_names(json_str):
        try:
            items = json.loads(json_str)
            return ", ".join([i["name"] for i in items[:4]])
        except:
            return ""

    df["genres_clean"]   = df["genres"].apply(parse_names)
    df["keywords_clean"] = df["keywords"].apply(parse_names)

    movies = []
    for _, row in df.iterrows():
        text = (
            f"Title: {row['title']}. "
            f"Genres: {row['genres_clean']}. "
            f"Keywords: {row['keywords_clean']}. "
            f"Overview: {row['overview']}. "
            f"Rating: {row['vote_average']}. "
            f"Year: {str(row['release_date'])[:4]}."
        )
        movies.append({
            "text": text,
            "metadata": {
                "title":  row["title"],
                "genres": row["genres_clean"],
                "year":   str(row["release_date"])[:4],
                "rating": str(row["vote_average"]),
            }
        })
    return movies
