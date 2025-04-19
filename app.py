import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Load dữ liệu & mô hình
df = pd.read_csv("data/cleaned_songs.csv")
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
tfidf_matrix = joblib.load("models/tfidf_matrix.pkl")

st.title("🎵 Song Recommendation System")
st.write("Gợi ý bài hát dựa trên tên bài hát hoặc lời bài hát bạn nhập.")

# Tabs: Tên bài hát hoặc Lyrics
tab1, tab2 = st.tabs(["🔍 Gợi ý theo tên bài hát", "📝 Gợi ý theo lời bài hát"])

with tab1:
    input_title = st.text_input("Nhập tên bài hát bạn yêu thích:")
    top_k = st.slider("Số bài hát gợi ý", 1, 10, 5, key="slider1")

    if st.button("Gợi ý bài hát", key="by_title"):
        from recommend import recommend_song
        if input_title:
            cosine_sim = joblib.load("models/cosine_sim.pkl")
            recommendations = recommend_song(input_title, df, cosine_sim, top_k=top_k)
            if isinstance(recommendations, str):
                st.warning(recommendations)
            else:
                st.success("🎧 Gợi ý bài hát:")
                sim_scores = cosine_sim[df[df['song'].str.lower() == input_title.lower()].index[0]]
                recommendations = recommendations.copy()
                recommendations["similarity (%)"] = [
                    f"{sim_scores[i]*100:.1f}%" for i in recommendations.index
                ]
                st.dataframe(recommendations.reset_index(drop=True))

        else:
            st.info("Vui lòng nhập tên bài hát!")

with tab2:
    input_lyrics = st.text_area("Nhập một đoạn lời bài hát:")
    top_k = st.slider("Số bài hát gợi ý", 1, 10, 5)

    if st.button("Gợi ý bài hát", key="by_lyrics"):
        if input_lyrics.strip():
            input_vec = tfidf_vectorizer.transform([input_lyrics])
            sim_scores = cosine_similarity(input_vec, tfidf_matrix).flatten()
            sim_indices = sim_scores.argsort()[::-1][:top_k]

            seen = set()
            results = []
            for i in sim_indices:
                title = df.iloc[i]['song'].strip().lower()
                artist = df.iloc[i]['artist'].strip().lower()
                key = (title, artist)
                if key not in seen:
                    seen.add(key)
                    results.append(i)
                if len(results) == top_k:
                    break

            if results:
                st.success("🎧 Các bài hát bạn có thể sẽ thích:")
                results_data = []
                for i in results:
                    sim = sim_scores[i] * 100
                    results_data.append({
                        "song": df.iloc[i]['song'],
                        "artist": df.iloc[i]['artist'],
                        "similarity (%)": f"{sim:.1f}%"
                    })

                st.dataframe(pd.DataFrame(results_data))

            else:
                st.warning("Không tìm thấy bài hát tương tự.")
        else:
            st.info("Vui lòng nhập lời bài hát!")
