import streamlit as st
import pandas as pd

# from test_script import get_preds
# from data_and_models import check_data, get_pred, convert_df, plot_shap_explanation
import numpy as np
import matplotlib.pyplot as plt

# from aux_functions import *
import os
import re
import datetime
import seaborn as sns
import matplotlib.pyplot as plt

import plotly.express as px
import torch
import seaborn as sns
import datetime

from collections import Counter
from wordcloud import WordCloud
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import DBSCAN, KMeans, Birch, AgglomerativeClustering
from tqdm import tqdm
from Levenshtein import distance as lev

from aux_functions import *





def analyze_hist_data():
    
    bank = st.radio( 'Выберите банк',
                        ['Тинькофф', 'Сбербанк', 'ВТБ', 'Альфа-банк']) 
    
    
    bank = map_bank[bank]
    
    
    df = pd.read_parquet(f'data/results_sravni_{bank}.parquet')
    df['category'] = df['category'].map(map_category)
    df['triplets'] = df['pred_text']
    df['timestamp'] = pd.to_datetime(df['time'].apply(lambda x: x[:10]))
    df = df[~df['text'].apply(lambda x: 'ЛовиОтзыв' in x)]

    count_dates = df['time'].value_counts()
    # count_dates[count_dates<100]
    df = df[df['time'].isin(count_dates[count_dates<100].index)]
    # st.write(df)
    # df


    # data['triplet'] = data['pred_text'].apply(lambda x: eval(x))
    # all_triplets = []
    # for label in data['triplet']:
    #     all_triplets.extend(label)
    # st.write(data)
    
    all_triplets = []
    all_times = []
    for _, (label, time) in df[['triplets', 'timestamp']].iterrows():
        # all_triplets.extend(eval(label))
        label = eval(label)
        all_triplets.extend(label)
        all_times += [time] * len(label)

    data = []
    for (triplet, time) in zip(all_triplets, all_times):
        # st.write(triplet)
        # st.write(triplet)
        at, ot, sp = triplet
        at = lemmatize(at.lower())
        ot = lemmatize(ot.lower())

        data.append([at, ot, sp, (at, ot, sp), (at, ot), time])
    data = pd.DataFrame(data)
    data.columns = ['aspect', 'opinion', 'sentiment', 'triplet', 'aspect_opinion', 'timestamp']
    data = data[~data.aspect.apply(is_stop_word)]
    
    aspect_counts = data['aspect'].value_counts()
    correct_aspects = {}
    for i, (a, count) in enumerate(aspect_counts.items()):
        if len(a) < 6: continue
        for j in range(i):
            b = aspect_counts.index[j]
            if a[0] == b[0] and a[-1] == b[-1] and lev(a, b) <= 1:
                correct_aspects[a] = b
                break
       
    data['aspect'] = data['aspect'].apply(lambda x : correct_aspects.get(x, x))
    

    st.write(df)
    
    # st.write(data)
    st.subheader("Cамые частые триплеты")

    def get_most_common_triplets(all_triplets, n):
        return [
            [as_, op, st, count]
            for (as_, op, st), count in Counter(data.triplet).most_common(n)
        ]

    most_common_triplets = get_most_common_triplets(all_triplets, 5)
    most_common_triplets = pd.DataFrame(most_common_triplets)
    most_common_triplets.columns = ["Аспект", "Мнение", "Тональность", "Частота"]
    st.write(most_common_triplets)
    
    def sentiment_hist(data, column):
        f, ax = plt.subplots(figsize=(8, 8))
        colors = {"NEG": "red", "POS": "green"}
        # f, ax = plt.subplots(figsize=(12, 12))
        sns.countplot(x="sentiment", data=data, hue="sentiment", palette=colors, ax=ax, order=['NEG', 'POS'])
        ax.set_xticks([0, 1], labels=["Негативные", "Положительные"])
        
        plt.title("Количество положительных и негативных триплетов")
        plt.xlabel("")
        plt.ylabel("Количество")
        plt.show()
        st.pyplot(f)
        
    # st.write(data)

    st.subheader("Гистограмма тональности")
    sentiment_hist(data, "sentiment")

    
    
    
    
    def common_sentiment_bytime(data):
        data_sorted = data.sort_values("timestamp", ascending=True)

        def count_pos_neg(group):
            sent_dict = {"count_pos": 0, "count_neg": 0}
            for item in group:
                if item == "POS":
                    sent_dict["count_pos"] += 1
                else:
                    sent_dict["count_neg"] += 1
            return sent_dict

        counted_sentiment = (
            data_sorted.groupby("timestamp")[["sentiment"]]
            .agg(count_pos_neg)
            .reset_index(names=["timestamp"])
        )

        counted_sentiment['pos'] = counted_sentiment['sentiment'].apply(lambda x: x['count_pos'])
        counted_sentiment['neg'] = counted_sentiment['sentiment'].apply(lambda x: x['count_neg'])
        
        window = 28 * 2
        sum_pos = counted_sentiment['pos'].rolling(window=window).sum().iloc[window:] + 1
        sum_neg = counted_sentiment['neg'].rolling(window=window).sum().iloc[window:] + 1
        ratio = sum_pos / sum_neg
        
        data_ratio = counted_sentiment[["timestamp"]]
        data_ratio["Доля положительных"] = ratio
        data_ratio['Банк'] = map_bank_reverse[bank]


        fig = px.line(data_ratio, x="timestamp", y="Доля положительных")
        fig.update_yaxes(range=[min(ratio) - 1, max(ratio) + 1], autorange=False)
        #   fig.get_axes().set_ylim(min(ratio)-1, max(ratio)+1)
        # fig.update_layout(scattermode="group")
        #   fig.yaxis_title('Доля положительных триплетов')
        #   fig.xaxis_title('Дата')
        fig.update_layout(
            title=f"Доля положительных триплетов по времени",
            xaxis_title="Дата",
            yaxis_title="Доля положительных триплетов",
        )
        # fig.show()
        st.plotly_chart(fig)

        
    st.subheader("Доля положительных тональностей с течением времени")
    common_sentiment_bytime(data)
    
    
    
    
    
    
    
    
    
    
    
    def num_reviews_bytime(data):
        data_sorted = data.sort_values("timestamp", ascending=True)
        
        data['timestamp'] = pd.to_datetime(data['time'].apply(lambda x: str(x)[:7]))
        data = pd.DataFrame(data['timestamp'].value_counts().sort_index()).reset_index()
        data.columns = ['timestamp', 'count']
        # st.write(data)


        fig = px.line(data, x='timestamp', y='count')
        fig.update_layout(
            title=f"Число отзывов по времени",
            xaxis_title="Дата",
            yaxis_title="Число отзывов",
        )

        st.plotly_chart(fig)

        
    st.subheader("Число отзывов с течением времени")
    num_reviews_bytime(df.copy())
    
    
    
    
    

    def aspect_sentiment_bytime(data, aspect):
        aspect_data = data.groupby("aspect").get_group(aspect).sort_values("timestamp")
        # st.write(aspect_data)

        def count_pos_neg(group):
            sent_dict = {"count_pos": 0, "count_neg": 0}
            for item in group:
                if item == "POS":
                    sent_dict["count_pos"] += 1
                else:
                    sent_dict["count_neg"] += 1
            return sent_dict

        counted_sentiment = (
            aspect_data.groupby("timestamp")[["sentiment"]]
            .agg(count_pos_neg)
            .reset_index(names=["timestamp"])
        )

        counted_sentiment['pos'] = counted_sentiment['sentiment'].apply(lambda x: x['count_pos'])
        counted_sentiment['neg'] = counted_sentiment['sentiment'].apply(lambda x: x['count_neg'])
        
        window = 14
        sum_pos = counted_sentiment['pos'].rolling(window=window).sum().iloc[window:] + 1
        sum_neg = counted_sentiment['neg'].rolling(window=window).sum().iloc[window:] + 1
        ratio = sum_pos / (sum_neg + sum_pos)
        
        data_ratio = counted_sentiment[["timestamp"]].iloc[window:]
        data_ratio["Доля положительных"] = ratio
        # data_ratio = data_ratio[((sum_neg + sum_pos) > 50).values]
        
        
        fig = px.line(data_ratio, x="timestamp", y="Доля положительных")
        fig.update_yaxes(range=[min(ratio) - 1, max(ratio) + 1], autorange=False)
        fig.update_layout(
            title=f'Зависимость доли положительных триплетов, Аспект - "{aspect}"',
            xaxis_title="Дата",
            yaxis_title="Доля положительных триплетов",
        )
        fig.update_yaxes(range=[0, 1], autorange=False)
        # fig.update_layout(scattermode="group")
        # fig.show()
        st.plotly_chart(fig)
    
    st.subheader("Тональность по аспектам с течением времени")
    possible_aspect = data['aspect'].value_counts()
    possible_aspect = possible_aspect[possible_aspect>=3]
    aspect = st.selectbox('Выберите аспект', possible_aspect.index, index=0)

    # aspect_sentiment_bytime(data, "изучение")
    aspect_sentiment_bytime(data, aspect)
    
    
    

    
    
    
    def plot_mean_ratings(data):
        fig, ax = plt.subplots(layout="constrained")
        result = (
            data
            .groupby("category")['rating']
            .agg('mean')
            .sort_values(ascending=False)
        )
        sns.barplot(y=result.index, x=result.values, orient='h')
        # plt.xticks(fontsize=10, rotation=45)
        ax.set_ylabel("Категория")
        ax.set_xlabel("Средняя оценка")
        ax.bar_label(ax.containers[0], fmt=lambda x: str(round(x, 2)), label_type='center')
        st.pyplot(fig)

    st.subheader("Средняя оценка по категории")
    plot_mean_ratings(df)
    
    
    
    
    
    
    
    
    
    
    
    def category_ratings_bytime(data, category):
        category_data = data.groupby("category").get_group(category).sort_values("timestamp")

        
        # st.write(category_data.groupby("timestamp")["rating"])
        
        # counted_sentiment = (
        #     category_data.groupby("timestamp")[["rating"]]
        #     .reset_index(names=["timestamp"])
        # )
        
        # counted_sentiment = (
        #     category_data.groupby("timestamp")[["rating"]]
        # )
        mean_ratings = category_data.groupby("timestamp")["rating"].apply(lambda x: sum(x)/len(x)).reset_index()
        # st.write(mean_ratings)
        
        window = 28
        mean_ratings['rating'] = mean_ratings['rating'].rolling(window=window).mean()
        mean_ratings = mean_ratings.iloc[window:]
        

        # data_ratio["Доля положительных"] = ratio[::smoothing_k]

        fig = px.line(mean_ratings, x="timestamp", y="rating")
        # fig.update_yaxes(range=[min(ratio) - 1, max(ratio) + 1], autorange=False)
        fig.update_layout(
            title=f'Зависимость среднего рейтинга по времени, Категория - "{category}"',
            xaxis_title="Дата",
            yaxis_title="Средний рейтинг",
        )
        # fig.update_layout(scattermode="group")
        # fig.show()
        st.plotly_chart(fig)
    
    st.subheader("Рейтинг по категориям с течением времени")
    possible_cats = df['category'].value_counts().index
    category = st.selectbox('Выберите категорию', possible_cats, index=0)

    # aspect_sentiment_bytime(data, "изучение")
    category_ratings_bytime(df, category)
    
    
    
    
    
    
    
    
    # 1/0

    def get_most_neg_aspects(data, n):
        fig, ax = plt.subplots(layout="constrained")
        result = (
            data[data.sentiment == "NEG"]
            .groupby("aspect")
            .size()
            .sort_values(ascending=False)
            .iloc[:n]
        )
        sns.barplot(x=result.index, y=result.values)
        ax.set_xlabel("Аспект")
        ax.set_ylabel("Кол-во негативных триплетов")
        ax.set_title("Cамые негативные аспекты")
        plt.xticks(fontsize=12, rotation=30)
        # plt.yticks(list(range(max(result.values) + 1)), fontsize=12)
        plt.yticks(fontsize=12)
        st.pyplot(fig)

    st.subheader("Гистограмма самых негативных аспектов")
    get_most_neg_aspects(data, 10)

    def get_most_pos_aspects(data, n):
        fig, ax = plt.subplots(layout="constrained")
        result = (
            data[data.sentiment == "POS"]
            .groupby("aspect")
            .size()
            .sort_values(ascending=False)
            .iloc[:10]
        )
        sns.barplot(x=result.index, y=result.values)
        plt.xticks(fontsize=10, rotation=30)
        ax.set_xlabel("Аспект")
        ax.set_ylabel("Кол-во положительных триплетов")
        ax.set_title("Самые позитивные аспекты")
        st.pyplot(fig)

    st.subheader("Гистограмма самых положительных аспектов")
    get_most_pos_aspects(data, 10)
    
    
    
    
    
    def get_part_of_pos_sentiments(x):
        if len(x) >= 200:
            return len(x[x=='POS'])/len(x)
        else:
            return None
        
    
    def plot_most_positive_aspects(data, n):
        fig, ax = plt.subplots(layout="constrained")
        most_positive = data.groupby('aspect').agg({'sentiment': get_part_of_pos_sentiments})\
                                                   ['sentiment'].sort_values(ascending=False).iloc[:10].to_dict()
        most_positive = {k:round(100*v, 2) for k, v in most_positive.items()}
        most_positive = {k:v for k, v in most_positive.items() if not pd.isna(v)}
        # st.write(most_positive.values())
        # st.write(most_positive)
        # st.write(most_positive.keys(), most_positive.values())
        sns.barplot(x=list(most_positive.keys()), y=list(most_positive.values()))
        

        plt.title('Cамые положительные аспекты')
        plt.ylabel('Доля положительных триплетов, %')
        plt.xlabel('Аспект')

        ax.set_ylim([min(most_positive.values())-1, 100])
        plt.xticks(fontsize=9, rotation=45)
        st.pyplot(fig);
    # plt.yticks(ticks=list(most_negative.values()),
    #     labels=[str(v) + '%' for v in list(most_negative.values())]);

    st.subheader("Аспекты с самой высокой долей положительных триплетов")
    plot_most_positive_aspects(data, 10)
    
    
    
    
    def get_part_of_neg_sentiments(x):
        # st.write(len(x[x=='Neg']))
        if len(x) >= 50:
            return len(x[x=='NEG'])/len(x)
        else:
            return None
        
    
    def plot_most_negative_aspects(data, n):
        fig, ax = plt.subplots(layout="constrained")
        most_negative = data.groupby('aspect').agg({'sentiment': get_part_of_neg_sentiments})\
                                                   ['sentiment'].sort_values(ascending=False).iloc[:10].to_dict()
        most_negative = {k:round(100*v, 2) for k, v in most_negative.items()}
        most_negative = {k:v for k, v in most_negative.items() if not pd.isna(v)}
        # st.write(most_negative.values())
        # st.write(data.groupby('aspect').agg({'sentiment': get_part_of_neg_sentiments})\
        #                                            ['sentiment'].sort_values(ascending=False).to_dict())
        # st.write(most_negative)
        # st.write(most_negative.keys(), most_negative.values())
        sns.barplot(x=list(most_negative.keys()), y=list(most_negative.values()))
        

        plt.title('Cамые негативные аспекты')
        plt.ylabel('Доля негативных триплетов, %')
        plt.xlabel('Аспект')

        ax.set_ylim([min(most_negative.values())-1, 100])
        plt.xticks(fontsize=9, rotation=45)
        st.pyplot(fig);
    # plt.yticks(ticks=list(most_negative.values()),
    #     labels=[str(v) + '%' for v in list(most_negative.values())]);

    st.subheader("Аспекты с самой высокой долей негативных триплетов")
    plot_most_negative_aspects(data, 10)
    
    
    

    def get_most_popular_aspect_opinion(data, n="all"):
        fig, ax = plt.subplots()
        data["aspect-opinion"] = data["aspect_opinion"].apply(
            lambda x: x[0] + "-" + x[1]
        )
        most_common_pairs = pd.DataFrame(
            Counter(data["aspect-opinion"]).most_common(10), columns=["pair", "count"]
        )
        most_common_pairs_series = pd.Series(
            list(most_common_pairs["count"]), index=list(most_common_pairs["pair"])
        )[:10]
        fig, ax = plt.subplots(layout="constrained")
        #   ax.bar(x=most_common_pairs_series.index, height=most_common_pairs_series.values)
        sns.barplot(
            x=most_common_pairs_series.index, y=most_common_pairs_series.values, ax=ax
        )
        plt.xticks(rotation=75)
        ax.set_xlabel("Аспект-мнение")
        ax.set_ylabel("Встречаемость")
        ax.set_title("Cамые частые пары аспект-мнение")
        st.pyplot(fig)

    st.subheader("Гистограмма популярности пар аспект-мнение")
    get_most_popular_aspect_opinion(data, 12)

    # st.write(data['aspect'].value_counts())
    # st.write(pd.Series((" ".join(data["aspect"])).split()).value_counts())
    def simple_wordcloud_aspect(data):
        fig, ax = plt.subplots()
        text=" ".join(data["aspect"])#.replace(' не ', ' не')
        # st.write(text)
        
        cloud_aspect = WordCloud(
            background_color="#ffffff",
            contour_width=20,
            contour_color="#2e3043",
            colormap="Set2",
            max_words=20,
            collocations=False
        ).generate(text=text)
        plt.imshow(cloud_aspect)
        plt.axis("off")
        plt.show()
        st.pyplot(fig)

    st.subheader("Облака слов для аспектов и мнений")
    simple_wordcloud_aspect(data)

    def simple_wordcloud_opinion(data):
        fig, ax = plt.subplots()
        text=" ".join(data["opinion"]).replace(' не ', ' не')
        cloud_opinion = WordCloud(background_color="#ffffff", max_words=20, collocations=False).generate(
            text=text
        )
        plt.imshow(cloud_opinion)
        plt.axis("off")
        plt.show()
        st.pyplot(fig)

    simple_wordcloud_opinion(data)

    from transformers import pipeline

    def plot_emotions(data):
        eng_to_rus = {
            "admiration": "восхищение",
            "amusement": "веселье",
            "anger": "злость",
            "annoyance": "раздражение",
            "approval": "одобрение",
            "caring": "забота",
            "confusion": "непонимание",
            "curiosity": "любопытство",
            "desire": "желание",
            "disappointment": "разочарование",
            "disapproval": "неодобрение",
            "disgust": "отвращение",
            "embarrassment": "смущение",
            "excitement": "возбуждение",
            "fear": "страх",
            "gratitude": "признательность",
            "grief": "горе",
            "joy": "радость",
            "love": "любовь",
            "nervousness": "нервозность",
            "optimism": "оптимизм",
            "pride": "гордость",
            "realization": "осознание",
            "relief": "облегчение",
            "remorse": "раскаяние",
            "sadness": "грусть",
            "surprise": "удивление",
            "neutral": "нейтральность",
        }

        fig, ax = plt.subplots(figsize=(9, 9))
        plt.title("Эмоции в отзывах студентов")
        plt.ylabel("Встречаемость эмоции")
        plt.xlabel("Эмоция")
        sns.countplot(x=data["emotion"])
        plt.xticks(
            list(range(len(data["emotion"].value_counts().index))),
            labels=[eng_to_rus[i] for i in data["emotion"].value_counts().index],
            fontsize=12,
            rotation=30,
        )
        st.pyplot(fig)

    # st.subheader("Эмоции студентов")
    # plot_emotions(pd.read_parquet("data/tink.parquet"))

    import umap

    def cluster_aspects(data):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(
            "cointegrated/rubert-tiny-sentiment-balanced"
        )
        model = AutoModel.from_pretrained("cointegrated/rubert-tiny-sentiment-balanced")

        model.to(device)

        aspects = set(data.aspect.tolist())
        embeddings = []

        for aspect in tqdm(aspects):

            encoded_input = tokenizer(aspect, padding=False, return_tensors="pt").to(
                device
            )

            with torch.no_grad():
                model_output = model(**encoded_input)

            embeddings.append(model_output.last_hidden_state.mean(dim=1))

        embeddings = torch.stack(embeddings).cpu().numpy().squeeze(1)

        reducer8 = umap.UMAP(n_components=4)
        reducer2 = umap.UMAP(n_components=2)

        embeds_reduced8 = reducer8.fit_transform(embeddings)
        embeds_reduced2 = reducer2.fit_transform(embeddings)

        clusters = DBSCAN(n_jobs=-1, min_samples=1).fit(embeds_reduced8)
        df = pd.DataFrame(embeds_reduced2[:, 0], columns=["x"])
        df["y"] = embeds_reduced2[:, 1]
        df["color"] = clusters.labels_
        df["aspect"] = list(aspects)
        # df["size"] = df["aspect"].apply(lambda x: len(data[data.aspect == x]) / 10)

        return px.scatter(
            df, x="x", y="y", color=df["color"], hover_data={"text": df["aspect"]},
            color_continuous_scale=[
        '#A3ADF8',
        '#7D88FA',
        
    ],
        )

    # fig = cluster_aspects(data)
    # st.subheader("Кластеризация аспектов")
    # st.plotly_chart(fig)

    import umap

    def cluster_opinions(data):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(
            "cointegrated/rubert-tiny-sentiment-balanced"
        )
        model = AutoModel.from_pretrained("cointegrated/rubert-tiny-sentiment-balanced")

        model.to(device)

        opinions = set(data.opinion.tolist())
        embeddings = []

        for opinion in tqdm(opinions):

            encoded_input = tokenizer(opinion, padding=False, return_tensors="pt").to(
                device
            )

            with torch.no_grad():
                model_output = model(**encoded_input)

            embeddings.append(model_output.last_hidden_state.mean(dim=1))

        embeddings = torch.stack(embeddings).cpu().numpy().squeeze(1)

        reducer8 = umap.UMAP(n_components=8)
        reducer2 = umap.UMAP(n_components=2)

        embeds_reduced8 = reducer8.fit_transform(embeddings)
        embeds_reduced2 = reducer2.fit_transform(embeddings)

        clusters = DBSCAN(n_jobs=-1, min_samples=4).fit(embeds_reduced8)
        df = pd.DataFrame(embeds_reduced2[:, 0], columns=["x"])
        df["y"] = embeds_reduced2[:, 1]
        df["color"] = clusters.labels_
        df["opinion"] = list(opinions)
        # df["size"] = df["aspect"].apply(lambda x: len(data[data.aspect == x]) / 10)

        return px.scatter(
            df, x="x", y="y", color=df["color"], hover_data={"text": df["opinion"]},
            color_continuous_scale=[
        '#A3ADF8',
        '#7D88FA',
        
    ],
        )
    # fig = cluster_opinions(data)
    # st.plotly_chart(fig)

    
    
    
    
    # st.subheader("Основные объекты отзыва")
#     # colors = {1: 'green', 0: 'red'}
#     fig, ax = plt.subplots()
#     sns.countplot(x="obj", data=preds, ax=ax)
#     plt.title('Основной объект отзыва')
#     # plt.xlabel
#     plt.xticks([0, 1, 2], labels=['Вебинар', 'Программа', 'Преподаватель'])
#     plt.xlabel('')
#     plt.ylabel('Количество')
#     st.pyplot(fig)
#     # plt.show()
    

#     st.subheader("Релевантные и нерелевантные отзывы")
#     colors = {1: 'green', 0: 'red'}
#     fig, ax = plt.subplots()
#     sns.countplot(x="rel", data=preds, palette=colors, ax=ax)
#     plt.title('Количество релевантных и нерелевантных отзывов')
#     # plt.xlabel
#     plt.xticks([0, 1], labels=['Нерелевантные', 'Релевантные'])
#     plt.xlabel('')
#     plt.ylabel('Количество')
#     st.pyplot(fig)
#     # plt.show()

#     st.subheader("Положительные и негативные отзывы")
#     colors = {1: 'green', 0: 'red'}
#     fig, ax = plt.subplots()
#     sns.countplot(x="sent", data=preds, palette=colors, ax=ax)
#     plt.title('Количество положительных и негативных отзывов')
#     # plt.xlabel
#     plt.xticks([0, 1], labels=['Негативные', 'Положительные'])
#     # plt.xlabel('Тональность')
#     plt.xlabel('')
#     plt.ylabel('Количество')
#     st.pyplot(fig)
#     # plt.show()
    
    
    
    
    st.write("The end")

#     def sentiment_hist(data, column):
#         colors = {"POS": "green", "NEG": "red"}
#         f, ax = plt.subplots(figsize=(12, 12))
#         sns.countplot(x="sentiment", data=data, hue="sentiment", palette=colors, ax=ax)
#         plt.title("Количество положительных и негативных триплетов")
#         plt.xticks([0, 1], labels=["Негативные", "Положительные"])
#         plt.xlabel("")
#         plt.ylabel("Количество")
#         plt.show()
#         st.pyplot(f)