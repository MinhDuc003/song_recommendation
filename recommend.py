def recommend_song(input_title, df, cosine_sim, top_k=5):
    # Tìm chỉ số bài hát đầu vào
    idx = df[df['song'].str.lower().str.strip() == input_title.lower().strip()].index
    if len(idx) == 0:
        return f"❌ Bài hát '{input_title}' không có trong tập dữ liệu."

    idx = idx[0]
    sim_scores = cosine_sim[idx]
    
    # Đặt similarity với chính nó = -1 để loại bỏ
    sim_scores[idx] = -1

    # Lấy chỉ số bài hát có điểm tương đồng cao nhất
    sim_indices = sim_scores.argsort()[::-1]

    # Lọc trùng tên & nghệ sĩ
    seen_pairs = set()
    unique_recommendations = []

    for i in sim_indices:
        song = df.iloc[i]['song'].strip().lower()
        artist = df.iloc[i]['artist'].strip().lower()
        pair = (song, artist)
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            unique_recommendations.append(i)
        if len(unique_recommendations) == top_k:
            break

    if not unique_recommendations:
        return "⚠️ Không tìm thấy bài hát tương tự."

    return df[['song', 'artist']].iloc[unique_recommendations].reset_index(drop=True)
