import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import numpy as np
df = pd.read_csv("spotify_preprocess.csv")

# 拆分 embedding 列为 10 个浮点列
df_embed = df['Embedding_10d'].str.split(',', expand=True).astype(float)
print(df_embed.iloc[:2])

# 数值特征
features_num = [
    'Danceability','Energy','Valence',
    'Loudness','Speechiness','Acousticness','Instrumentalness'
]

# 提取这些列并拼接
X_numeric = df[features_num].astype(float).reset_index(drop=True)
X = pd.concat([X_numeric, df_embed.reset_index(drop=True)], axis=1).values


# set id2index list
id_to_index = {}
for i,id in enumerate(df["id"]):
    id_to_index[id] = i


def imple_recommend(id,k=5,mode="cosine"):
    q_index = id_to_index[id]
    q_vec = X[q_index].reshape(1, -1)

    if mode == "cosine":
        sim_scores = cosine_similarity(q_vec, X).flatten()
        sim_scores[q_index] = -1  # 排除自己
        top_k = sim_scores.argsort()[-k:][::-1]   #返回的是从大到小的位置索引列表
        print(f"✅ 使用余弦相似度推荐 Top-{k}：")
        print(top_k)

    elif mode == "euclidean":
        dist_scores = euclidean_distances(q_vec, X).flatten()
        dist_scores[q_index] = np.inf
        top_k = dist_scores.argsort()[:k]
        print(f"✅ 使用欧氏距离推荐 Top-{k}：")
        print(top_k)
    else:
        raise ValueError("mode must be 'cosine' or 'euclidean'")
    
    for i, idx in enumerate(top_k):
            song_id = df.iloc[idx]['id']
            score = sim_scores[idx] if mode == "cosine" else dist_scores[idx]
            print(f"Top {i+1}:  ID: {song_id} | Score: {score:.4f}")

imple_recommend("003vvx7Niy0yvhvHt4a68B")
    
    