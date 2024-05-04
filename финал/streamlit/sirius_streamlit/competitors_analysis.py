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

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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


def analyze_competitors():
    
    bank = st.radio( 'Выберите основной банк',
                        ['Тинькофф', 'Сбербанк', 'ВТБ', 'Альфа-банк']) 
    
    possible_competitors = ['Тинькофф', 'Сбербанк', 'ВТБ', 'Альфа-банк']
    possible_competitors.remove(bank)
    
    competitors = st.multiselect( 'Выберите конкурента/ов для анализа',
                        possible_competitors, possible_competitors) 
    
    
    

    
    bank = map_bank[bank]
    competitors = [map_bank[competitor] for competitor in competitors]
    
    n_competitors = len(competitors)
    
    
    
    df = pd.read_parquet(f'data/results_sravni_{bank}.parquet')
    df['category'] = df['category'].map(map_category)
    df['triplets'] = df['pred_text']
    df['timestamp'] = pd.to_datetime(df['time'].apply(lambda x: x[:10]))
    df = df[~df['text'].apply(lambda x: 'ЛовиОтзыв' in x)]

    count_dates = df['time'].value_counts()
    df = df[df['time'].isin(count_dates[count_dates<100].index)]
    
    
    
    
    
    
    
    
    df_competitors = []
    for competitor in competitors:
        df_competitor = pd.read_parquet(f'data/results_sravni_{competitor}.parquet')
        df_competitor['category'] = df_competitor['category'].map(map_category)
        df_competitor['triplets'] = df_competitor['pred_text']
        df_competitor['timestamp'] = pd.to_datetime(df_competitor['time'].apply(lambda x: x[:10]))
        df_competitor = df_competitor[~df_competitor['text'].apply(lambda x: 'ЛовиОтзыв' in x)]

        count_dates = df_competitor['time'].value_counts()
        df_competitor = df_competitor[df_competitor['time'].isin(count_dates[count_dates<100].index)]
    
    
        df_competitors.append(df_competitor)
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
    
    
    
    
    all_triplets_competitors = []
    all_times_competitors = []
    for i in range(n_competitors):
        all_triplets = []
        all_times = []
        for _, (label, time) in df_competitors[i][['triplets', 'timestamp']].iterrows():
            # all_triplets.extend(eval(label))
            label = eval(label)
            all_triplets.extend(label)
            all_times += [time] * len(label)
        all_triplets_competitors.append(all_triplets)
        all_times_competitors.append(all_times)
        
        
    data_competitors = []
    for i in range(n_competitors):
        data_competitor = []
        for (triplet, time) in zip(all_triplets_competitors[i], all_times_competitors[i]):
            at, ot, sp = triplet
            at = lemmatize(at.lower())
            ot = lemmatize(ot.lower())
            data_competitor.append([at, ot, sp, (at, ot, sp), (at, ot), time])
        

        
        
        data_competitor = pd.DataFrame(data_competitor)
        data_competitor.columns = ['aspect', 'opinion', 'sentiment', 'triplet', 'aspect_opinion', 'timestamp']
        data_competitor = data_competitor[~data_competitor.aspect.apply(is_stop_word)]

        aspect_counts = data_competitor['aspect'].value_counts()
        correct_aspects = {}
        for i, (a, count) in enumerate(aspect_counts.items()):
            if len(a) < 6: continue
            for j in range(i):
                b = aspect_counts.index[j]
                if a[0] == b[0] and a[-1] == b[-1] and lev(a, b) <= 1:
                    correct_aspects[a] = b
                    break

        data_competitor['aspect'] = data_competitor['aspect'].apply(lambda x : correct_aspects.get(x, x))
        
        data_competitors.append(data_competitor)

    
    
    
    
    
    
    
    
    
    
    def sentiment_hist(data, column, fig, ax, bank):
        colors = {"NEG": "red", "POS": "green"}
        # f, ax = plt.subplots(figsize=(12, 12))
        sns.countplot(x="sentiment", data=data, hue="sentiment", palette=colors, ax=ax, order=['NEG', 'POS'])
        ax.set_xticks([0, 1], labels=["Негативные", "Положительные"])
        ax.set_xlabel("")
        ax.set_ylabel("Количество")
        # ax.get_legend().remove()
        ax.set_title(map_bank_eng_to_rus[bank], fontsize=12)
        # plt.show()
        # st.pyplot(fig)
        
    # st.write(data)

    
    nrows, ncols = (n_competitors + 1)//2 + (n_competitors + 1)%2, 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    fig.suptitle("Количество положительных и негативных триплетов", fontsize=18)
    
    axis_generator =  generate_axis(nrows, ncols, n_competitors, axes)
    
    
    for i, ax in enumerate(axis_generator):
        if i == 0:
            st.write(data)
            sentiment_hist(data, "sentiment", fig, ax, bank)

        else:
            sentiment_hist(data_competitors[i-1], "sentiment", fig, ax, competitors[i-1])
    
    st.subheader("Гистограмма тональности")
    st.pyplot(fig)
    
    
    
    
    data = data[~data.aspect.apply(is_stop_word)]
    for i, data_competitor in enumerate(data_competitors):
        data_competitors[i] = data_competitor[~data_competitor.aspect.apply(is_stop_word)]
    
    
    
    
    
    
    
    
#     def common_sentiment_bytime(data, bank):
#         data_sorted = data.sort_values("timestamp", ascending=True)

#         def count_pos_neg(group):
#             sent_dict = {"count_pos": 0, "count_neg": 0}
#             for item in group:
#                 if item == "POS":
#                     sent_dict["count_pos"] += 1
#                 else:
#                     sent_dict["count_neg"] += 1
#             return sent_dict

#         counted_sentiment = (
#             data_sorted.groupby("timestamp")[["sentiment"]]
#             .agg(count_pos_neg)
#             .reset_index(names=["timestamp"])
#         )

#         count_pos = 1
#         count_neg = 1
#         ratio = []
#         for i, item in counted_sentiment.iterrows():
#             count_pos += item["sentiment"]["count_pos"]
#             count_neg += item["sentiment"]["count_neg"]
#             # ratio.append(count_pos / count_neg )
#             ratio.append(100 * count_pos / (count_neg + count_pos))

#         smoothing_k = 2
#         data_ratio = counted_sentiment[["timestamp"]].iloc[::smoothing_k]
#         data_ratio["Доля положительных"] = ratio[::smoothing_k]

#         fig = px.line(data_ratio, x="timestamp", y="Доля положительных")
#         fig.update_yaxes(range=[min(ratio) - 1, max(ratio) + 1], autorange=False)
#         #   fig.get_axes().set_ylim(min(ratio)-1, max(ratio)+1)
#         # fig.update_layout(scattermode="group")
#         #   fig.yaxis_title('Доля положительных триплетов')
#         #   fig.xaxis_title('Дата')
#         fig.update_layout(
#             # title=f"Доля положительных триплетов по времени",
#             title=map_bank_eng_to_rus[bank],
#             xaxis_title="Дата",
#             yaxis_title="Доля положительных триплетов",
#         )
#         # fig.show()
#         st.plotly_chart(fig)

#     st.subheader("Доля положительных тональностей с течением времени")
    
    
#     for i in range(n_competitors + 1):
#         if i == 0:
#             common_sentiment_bytime(data, bank)
#         else:
#             common_sentiment_bytime(data_competitors[i-1], competitors[i-1])
            
    
    def common_sentiment_bytime(data, data_competitors):
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
        
        window = 28
        sum_pos = counted_sentiment['pos'].rolling(window=window).sum().iloc[window:] + 1
        sum_neg = counted_sentiment['neg'].rolling(window=window).sum().iloc[window:] + 1
        ratio = sum_pos / sum_neg
        
        # st.write(counted_sentiment)

        # for i, item in counted_sentiment.iterrows():
        #     # count_pos += item["sentiment"]["count_pos"]
        #     # count_neg += item["sentiment"]["count_neg"]
        #     count_pos = item["sentiment"]["count_pos"]
        #     count_neg = item["sentiment"]["count_neg"]
        #     # ratio.append(count_pos / count_neg )
        #     ratio.append(100 * count_pos / (count_neg + count_pos))

        data_ratio = counted_sentiment[["timestamp"]]
        data_ratio["Доля положительных"] = ratio
        data_ratio['Банк'] = map_bank_reverse[bank]
        data_ratio_all = data_ratio.copy()


        for i in range(n_competitors):
            data_sorted = data_competitors[i].sort_values("timestamp", ascending=True)

            counted_sentiment = (
                data_sorted.groupby("timestamp")[["sentiment"]]
                .agg(count_pos_neg)
                .reset_index(names=["timestamp"])
            )

            counted_sentiment['pos'] = counted_sentiment['sentiment'].apply(lambda x: x['count_pos'])
            counted_sentiment['neg'] = counted_sentiment['sentiment'].apply(lambda x: x['count_neg'])

            window = 28
            sum_pos = counted_sentiment['pos'].rolling(window=window).sum().iloc[window:]
            sum_neg = counted_sentiment['neg'].rolling(window=window).sum().iloc[window:]
            ratio = sum_pos / (sum_neg + sum_pos)
        
            data_ratio = counted_sentiment[["timestamp"]].iloc[window:]
            data_ratio["Доля положительных"] = ratio
            data_ratio['Банк'] = map_bank_reverse[competitors[i]]
            data_ratio_all = pd.concat([data_ratio_all, data_ratio], axis=0)
    
    
    
    
    
        fig = px.line(data_ratio_all, x="timestamp", y="Доля положительных", color='Банк',
             color_discrete_map={map_bank_reverse[bank]:color for bank, color in colors.items()})
        # fig.update_yaxes(range=[min(ratio) - 1, max(ratio) + 1], autorange=False)


        fig.update_layout(
            title=f"Доля положительных триплетов по времени",
            # title=map_bank_eng_to_rus[bank],
            xaxis_title="Дата",
            yaxis_title="Доля положительных триплетов",
        )
        # fig.show()
        st.plotly_chart(fig)
        
        
    st.subheader("Доля положительных тональностей с течением времени")
    common_sentiment_bytime(data, data_competitors)
    
    
    
    
    
    
    
    
    
    
#     def num_reviews_bytime(data, bank):
#         data_sorted = data.sort_values("timestamp", ascending=True)
        
#         data['timestamp'] = pd.to_datetime(data['time'].apply(lambda x: str(x)[:7]))
#         data = pd.DataFrame(data['timestamp'].value_counts().sort_index()).reset_index()
#         data.columns = ['timestamp', 'count']
#         # st.write(data)


#         fig = px.line(data, x='timestamp', y='count')
#         fig.update_layout(
#             title=map_bank_eng_to_rus[bank],
#             xaxis_title="Дата",
#             yaxis_title="Число отзывов",
#         )

#         st.plotly_chart(fig)

        
#     st.subheader("Число отзывов с течением времени")
#     for i in range(n_competitors + 1):
#         if i == 0:
#             num_reviews_bytime(df.copy(), bank)
#         else:
#             num_reviews_bytime(df_competitors[i-1], competitors[i-1])

    
    
    def num_reviews_bytime(data, data_competitors):
        
        data['timestamp'] = pd.to_datetime(data['time'].apply(lambda x: str(x)[:7]))
        data = pd.DataFrame(data['timestamp'].value_counts().sort_index()).reset_index()
        data.columns = ['timestamp', 'count']
        data['Банк'] = map_bank_reverse[bank]
        # st.write(competitors)
        # st.write(colors)
        # st.write(data)
        
        for i in range(n_competitors):
            data_competitor = data_competitors[i]
            data_competitor['timestamp'] = pd.to_datetime(data_competitor['time'].apply(lambda x: str(x)[:7]))
            data_competitor = pd.DataFrame(data_competitor['timestamp'].value_counts().sort_index()).reset_index()
            data_competitor.columns = ['timestamp', 'count']
            data_competitor['Банк'] = map_bank_reverse[competitors[i]]
            data = pd.concat([data, data_competitor], axis=0)
            
        
            
        fig = px.line(data, x='timestamp', y='count', color='Банк', color_discrete_map={map_bank_reverse[bank]:color for bank, color in colors.items()})
        fig.update_layout(
            # title=map_bank_eng_to_rus[bank],
            xaxis_title="Дата",
            yaxis_title="Число отзывов"
        )

        st.plotly_chart(fig)


      
    
    
        
    st.subheader("Число отзывов с течением времени")
    num_reviews_bytime(df.copy(), df_competitors.copy())
    
    
    
    
    
    
    
    
    def aspect_sentiment_bytime(data, data_competitors, aspect):
        aspect_data = data.groupby("aspect").get_group(aspect).sort_values("timestamp")
        mean_sent = aspect_data['sentiment'].map({'POS':1, 'NEG':0}).mean()


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
        
        window = 28
        sum_pos = counted_sentiment['pos'].rolling(window=window).sum().iloc[window:] + 1
        sum_neg = counted_sentiment['neg'].rolling(window=window).sum().iloc[window:] + 1
        ratio = sum_pos / (sum_neg + sum_pos)
        
        
        data_ratio = counted_sentiment[["timestamp"]]
        data_ratio["Доля положительных"] = ratio
        data_ratio['Банк'] = map_bank_reverse[bank]
        data_ratio_all = data_ratio[window:].copy()
        st.write("Банк - {}, Доля положительной тональности {:.2f}%, Всего триплетов - {}".format(map_bank_reverse[bank], 100*mean_sent, data[data.aspect == aspect].shape[0]))
        
        
        for i in range(n_competitors):
            aspect_data = data_competitors[i].groupby("aspect").get_group(aspect).sort_values("timestamp")
            mean_sent = aspect_data['sentiment'].map({'POS':1, 'NEG':0}).mean()

            counted_sentiment = (
                aspect_data.groupby("timestamp")[["sentiment"]]
                .agg(count_pos_neg)
                .reset_index(names=["timestamp"])
            )

            counted_sentiment['pos'] = counted_sentiment['sentiment'].apply(lambda x: x['count_pos'])
            counted_sentiment['neg'] = counted_sentiment['sentiment'].apply(lambda x: x['count_neg'])

            window = 28
            sum_pos = counted_sentiment['pos'].rolling(window=window).sum().iloc[window:]
            sum_neg = counted_sentiment['neg'].rolling(window=window).sum().iloc[window:]
            ratio = sum_pos / (sum_neg + sum_pos)
            
            st.write("Банк - {}, Доля положительной тональности {:.2f}%, Всего триплетов - {}".format(map_bank_reverse[competitors[i]], 100*mean_sent, data_competitors[i][data_competitors[i].aspect == aspect].shape[0]))
        
            data_ratio = counted_sentiment[["timestamp"]].iloc[window:]
            data_ratio["Доля положительных"] = ratio
            data_ratio['Банк'] = map_bank_reverse[competitors[i]]
            data_ratio_all = pd.concat([data_ratio_all, data_ratio[window:]], axis=0)
        
            
            
        fig = px.line(data_ratio_all, x="timestamp", y="Доля положительных", color='Банк',
             color_discrete_map={map_bank_reverse[bank]:color for bank, color in colors.items()})
        # fig.update_yaxes(range=[min(ratio) - 1, max(ratio) + 1], autorange=False)


        fig.update_layout(
            title=f"Доля положительных триплетов по времени",
            # title=map_bank_eng_to_rus[bank],
            xaxis_title="Дата",
            yaxis_title="Доля положительных триплетов",
        )
        # fig.show()
        st.plotly_chart(fig)
        
        

#         fig = px.line(data_ratio, x="timestamp", y="Доля положительных")
#         fig.update_yaxes(range=[min(ratio) - 1, max(ratio) + 1], autorange=False)
#         fig.update_layout(
#             title=f'Доля положительных триплетов, Аспект - "{aspect}", Банк - "{map_bank_eng_to_rus[bank]}"',
#             xaxis_title="Дата",
#             yaxis_title="Доля положительных триплетов",
#         )
#         # fig.update_layout(scattermode="group")
#         # fig.show()
        # st.plotly_chart(fig)
    
    st.subheader("Тональность по аспектам с течением времени")
    possible_aspect = data['aspect'].value_counts()
    possible_aspect = possible_aspect[possible_aspect>=3]
    aspect = st.selectbox('Выберите аспект', possible_aspect.index, index=0)


    aspect_sentiment_bytime(data.copy(), data_competitors.copy(), aspect)
    
    
    
    
    
    
    
    
#     def plot_mean_ratings(data):
#         fig, ax = plt.subplots(layout="constrained")
#         # result = (
#         #     data
#         #     .groupby("category")['rating']
#         #     .agg('mean')
#         #     .sort_values(ascending=False)
#         # )
#         # sns.barplot(y=result.index, x=result.values, orient='h')
#         # # plt.xticks(fontsize=10, rotation=45)
#         # ax.set_ylabel("Категория")
#         # ax.set_xlabel("Средняя оценка")
#         # ax.bar_label(ax.containers[0], fmt=lambda x: str(round(x, 2)), label_type='center')
#         # st.pyplot(fig)
#         mean_ratings = []

#         mean_rating = df.groupby("category")['rating'] \
#                 .agg('mean') \
#                 .sort_values(ascending=False)
#         mean_rating = pd.DataFrame(mean_rating)
#         mean_rating['bank'] = bank

#         mean_ratings.append(mean_rating)

#         for i in range(n_competitors):
#             mean_rating = df_competitors[i].groupby("category")['rating'] \
#                     .agg('mean') \
#                     .sort_values(ascending=False)
#             mean_rating = pd.DataFrame(mean_rating)
#             mean_rating['bank'] = competitors[i]
#             mean_ratings.append(mean_rating)

#         mean_ratings
#         mean_ratings = pd.DataFrame(pd.concat(mean_ratings, axis=0)).reset_index()
#         # mean_ratings = mean_ratings.T
#         # mean_ratings['bank'] = ([bank] + competitors
#         mean_ratings['bank'] = mean_ratings['bank'].map(map_bank_reverse)
        
        
#         for cat in np.unique(mean_ratings['category']):
#             fig, ax = plt.subplots(layout="constrained", figsize=(7, 7))
#             sns.barplot(mean_ratings[mean_ratings.category==cat], x='bank', y='rating', ax=ax)
#             ax.bar_label(ax.containers[0], fmt=lambda x: str(round(x, 2)), label_type='edge')
#             ax.set_title(cat, fontsize=12)
#             ax.set_ylabel('Средняя оценка')
#             ax.set_xlabel('Банк')
#             st.pyplot(fig);

    def plot_mean_ratings(data, cat):
        fig, ax = plt.subplots(layout="constrained")
        
        mean_ratings = []

        mean_rating = df.groupby("category")['rating'] \
                .agg('mean') \
                .sort_values(ascending=False)
        mean_rating = pd.DataFrame(mean_rating)
        mean_rating['bank'] = bank

        mean_ratings.append(mean_rating)

        for i in range(n_competitors):
            mean_rating = df_competitors[i].groupby("category")['rating'] \
                    .agg('mean') \
                    .sort_values(ascending=False)
            mean_rating = pd.DataFrame(mean_rating)
            mean_rating['bank'] = competitors[i]
            mean_ratings.append(mean_rating)

        mean_ratings
        mean_ratings = pd.DataFrame(pd.concat(mean_ratings, axis=0)).reset_index()
        # mean_ratings = mean_ratings.T
        # mean_ratings['bank'] = ([bank] + competitors
        mean_ratings['bank'] = mean_ratings['bank'].map(map_bank_reverse)
        
        
        
        # for cat in np.unique(mean_ratings['category']):
        fig, ax = plt.subplots(layout="constrained", figsize=(7, 7))
        sns.barplot(mean_ratings[mean_ratings.category==cat], x='bank', y='rating', ax=ax)
        ax.bar_label(ax.containers[0], fmt=lambda x: str(round(x, 2)), label_type='edge')
        ax.set_title(cat, fontsize=12)
        ax.set_ylabel('Средняя оценка')
        ax.set_xlabel('Банк')
        st.pyplot(fig);
           
            
    st.subheader("Средняя оценка по категории")
    
    possible_cats = df['category'].value_counts().index
    cat = st.selectbox('Выберите аспект', possible_cats, index=0)
    plot_mean_ratings(df, cat)
    
    
    
    
    
    
    
    
    
    
    
    def category_ratings_bytime(df, df_competitors, category):
        category_data = df.groupby("category").get_group(category).sort_values("timestamp")
     
        mean_ratings = category_data.groupby("timestamp")["rating"].apply(lambda x: sum(x)/len(x)).reset_index()
        # st.write(mean_ratings)
        
        window = 28
        mean_ratings['rating'] = mean_ratings['rating'].rolling(window=window).mean()
        mean_ratings = mean_ratings.iloc[window:]
        mean_ratings['Банк'] = map_bank_reverse[bank]
        mean_ratings_all = mean_ratings.copy()
        
        for i in range(n_competitors):
            
            category_data = df_competitors[i].groupby("category").get_group(category).sort_values("timestamp")
            mean_ratings = category_data.groupby("timestamp")["rating"].apply(lambda x: sum(x)/len(x)).reset_index()
            # st.write(mean_ratings)

            window = 28
            mean_ratings['rating'] = mean_ratings['rating'].rolling(window=window).mean()
            mean_ratings = mean_ratings.iloc[window:]
            mean_ratings['Банк'] = map_bank_reverse[competitors[i]]
            mean_ratings_all = pd.concat([mean_ratings_all, mean_ratings], axis=0)
            
        

        # data_ratio["Доля положительных"] = ratio[::smoothing_k]

        fig = px.line(mean_ratings_all, x="timestamp", y="rating", color='Банк',
                     color_discrete_map={map_bank_reverse[bank]:color for bank, color in colors.items()})
        # fig.update_yaxes(range=[min(ratio) - 1, max(ratio) + 1], autorange=False)
        fig.update_layout(
            title=f'Зависимость среднего рейтинга по времени, Категория - "{category}""',
            xaxis_title="Дата",
            yaxis_title="Средний рейтинг",
        )
        # fig.update_layout(scattermode="group")
        # fig.show()
        st.plotly_chart(fig)
    
    st.subheader("Рейтинг по категориям с течением времени")
    category = st.selectbox('Выберите категорию', possible_cats, index=0)

    category_ratings_bytime(df, df_competitors, category)
    # for i in range(n_competitors + 1):
    #     if i == 0:
    #         category_ratings_bytime(df, category, bank)
    #     else:
    #         category_ratings_bytime(df_competitors[i-1], category, competitors[i-1])
    
    
    
    
    
    
    
    
    
    
    
    def get_most_neg_aspects(data, n, fig, ax, bank):
        result = (
            data[data.sentiment == "NEG"]
            .groupby("aspect")
            .size()
            .sort_values(ascending=False)
            .iloc[:n]
        )
        sns.barplot(x=result.index, y=result.values, ax=ax)
        plt.sca(ax)
        plt.xticks(rotation=45)
        ax.set_xlabel("Аспект")
        ax.set_ylabel("Кол-во негативных триплетов")
        ax.set_title(map_bank_eng_to_rus[bank], fontsize=12)

    st.subheader("Гистограмма самых негативных аспектов")
    
    n = 10
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    fig.suptitle("Cамые негативные аспекты", fontsize=18)
    
    axis_generator =  generate_axis(nrows, ncols, n_competitors, axes)
    
    
    for i, ax in enumerate(axis_generator):
        if i == 0:
            get_most_neg_aspects(data, n, fig, ax, bank)

        else:
            get_most_neg_aspects(data_competitors[i-1], n, fig, ax, competitors[i-1])
    
    st.pyplot(fig)
    
    
    
    
    
    
    
    
    def get_part_of_neg_sentiments(x):
        # st.write(len(x[x=='Neg']))
        if len(x) >= 50:
            return len(x[x=='NEG'])/len(x)
        else:
            return None

        
    
    def plot_most_negative_aspects(data, n, fig, ax, bank):
        most_negative = data.groupby('aspect').agg({'sentiment': get_part_of_neg_sentiments})\
                                                   ['sentiment'].sort_values(ascending=False).iloc[:10].to_dict()
        most_negative = {k:round(100*v, 2) for k, v in most_negative.items()}
        most_negative = {k:v for k, v in most_negative.items() if not pd.isna(v)}
        # st.write(most_negative.values())
        # st.write(most_negative)
        # st.write(most_negative.keys(), most_negative.values())
        sns.barplot(x=list(most_negative.keys()), y=list(most_negative.values()), ax=ax)
        plt.sca(ax)
        

        ax.set_title(map_bank_eng_to_rus[bank], fontsize=12)
        ax.set_ylabel('Доля негативных триплетов, %')
        ax.set_xlabel('Аспект')

        # ax.set_ylim([min(most_negative.values())-1, 100])
        plt.xticks(fontsize=12, rotation=30)
        # st.pyplot(fig);
    # plt.yticks(ticks=list(most_negative.values()),
    #     labels=[str(v) + '%' for v in list(most_negative.values())]);
    
    n = 10
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 18))
    fig.suptitle("Аспекты с самой высокой долей негативных триплетов", fontsize=18)
    
    axis_generator =  generate_axis(nrows, ncols, n_competitors, axes)
    
    
    for i, ax in enumerate(axis_generator):
        if i == 0:
            plot_most_negative_aspects(data, n, fig, ax, bank)

        else:
            plot_most_negative_aspects(data_competitors[i-1], n, fig, ax, competitors[i-1])
    
    st.pyplot(fig)
    
    
    
    
    
    
    
    
    
    def get_most_pos_aspects(data, n, fig, ax, bank):
        result = (
            data[data.sentiment == "POS"]
            .groupby("aspect")
            .size()
            .sort_values(ascending=False)
            .iloc[:n]
        )
        sns.barplot(x=result.index, y=result.values, ax=ax)
        plt.sca(ax)
        plt.xticks(rotation=45)
        ax.set_xlabel("Аспект")
        ax.set_ylabel("Кол-во положительных триплетов")
        ax.set_title(map_bank_eng_to_rus[bank], fontsize=12)

    st.subheader("Гистограмма самых положительных аспектов")
    
    n = 10
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    fig.suptitle("Cамые положительные аспекты", fontsize=18)
    
    axis_generator =  generate_axis(nrows, ncols, n_competitors, axes)
    
    
    for i, ax in enumerate(axis_generator):
        if i == 0:
            get_most_pos_aspects(data, n, fig, ax, bank)

        else:
            get_most_pos_aspects(data_competitors[i-1], n, fig, ax, competitors[i-1])
    
    st.pyplot(fig)
    
    
    
    
    
    
    
    def get_part_of_pos_sentiments(x):
        if len(x) >= 100:
            return len(x[x=='POS'])/len(x)
        else:
            return None
        
    
    def plot_most_positive_aspects(data, n, fig, ax, bank):
        most_positive = data.groupby('aspect').agg({'sentiment': get_part_of_pos_sentiments})\
                                                   ['sentiment'].sort_values(ascending=False).iloc[:10].to_dict()
        most_positive = {k:round(100*v, 2) for k, v in most_positive.items()}
        most_positive = {k:v for k, v in most_positive.items() if not pd.isna(v)}
        # st.write(most_positive.values())
        # st.write(most_positive)
        # st.write(most_positive.keys(), most_positive.values())
        sns.barplot(x=list(most_positive.keys()), y=list(most_positive.values()), ax=ax)
        plt.sca(ax)
        

        ax.set_title(map_bank_eng_to_rus[bank], fontsize=12)
        ax.set_ylabel('Доля положительных триплетов, %')
        ax.set_xlabel('Аспект')

        # ax.set_ylim([min(most_positive.values())-1, 100])
        plt.xticks(fontsize=12, rotation=30)
        # st.pyplot(fig);
    # plt.yticks(ticks=list(most_negative.values()),
    #     labels=[str(v) + '%' for v in list(most_negative.values())]);
    
    n = 10
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18, 18))
    fig.suptitle("Аспекты с самой высокой долей положительных триплетов", fontsize=18)
    
    axis_generator =  generate_axis(nrows, ncols, n_competitors, axes)
    
    
    for i, ax in enumerate(axis_generator):
        if i == 0:
            plot_most_positive_aspects(data, n, fig, ax, bank)

        else:
            plot_most_positive_aspects(data_competitors[i-1], n, fig, ax, competitors[i-1])
    
    st.pyplot(fig)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def get_most_popular_aspect_opinion(data, n, fig, ax, bank):


        most_common_pairs = pd.DataFrame(
            Counter(data["aspect_opinion"].apply(
            lambda x: x[0] + "-" + x[1]
        )).most_common(10), columns=["pair", "count"]
        )
        most_common_pairs_series = pd.Series(
            list(most_common_pairs["count"]), index=list(most_common_pairs["pair"])
        )[:10]
        #   ax.bar(x=most_common_pairs_series.index, height=most_common_pairs_series.values)
        
        
        sns.barplot(
            y=most_common_pairs_series.index, x=most_common_pairs_series.values, ax=ax, orient='h'
        )
        
        ax.bar_label(ax.containers[0], labels=[tick.get_text() for tick in ax.get_yticklabels()], label_type='center', padding=30)
        plt.sca(ax)
        plt.yticks(ax.get_yticks(), [''] * len(ax.get_yticks()), rotation=90)
        
        
        
        # Get the current axes
        ax = plt.gca()

        
        ax.set_xlabel("Аспект-мнение")
        ax.set_ylabel("Встречаемость")
        ax.set_title(map_bank_eng_to_rus[bank], fontsize=12)
        
   
    
    st.subheader("Гистограмма популярности пар аспект-мнение")
    
    n = 12
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 14))
    fig.suptitle("Cамые частые пары аспект-мнение", fontsize=18)
    
    axis_generator =  generate_axis(nrows, ncols, n_competitors, axes)
    for i, ax in enumerate(axis_generator):
        if i == 0:
            get_most_popular_aspect_opinion(data, n, fig, ax, bank)

        else:
            get_most_popular_aspect_opinion(data_competitors[i-1], n, fig, ax, competitors[i-1])

    
    st.pyplot(fig)
    
    
    
    
    
    
    
    
    
    
    
    
#     def simple_wordcloud_aspect(data, fig, ax, bank):
#         fig, ax = plt.subplots()
#         text=" ".join(data["aspect"])#.replace(' не ', ' не')
        
#         cloud_aspect = WordCloud(
#             background_color="#ffffff",
#             contour_width=20,
#             contour_color="#2e3043",
#             colormap="Set2",
#             max_words=20,
#             collocations=False
#         ).generate(text=text)
#         plt.imshow(cloud_aspect)
#         plt.axis("off")
#         ax.set_title(map_bank_eng_to_rus[bank], fontsize=12)

#     st.subheader("Облака слов для аспектов и мнений")
    
 
    
#     fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
#     fig.suptitle("Аспекты", fontsize=18)
    
#     axis_generator =  generate_axis(nrows, ncols, n_competitors, axes)

    
#     for i, ax in enumerate(axis_generator):
#         if i == 0:
#             simple_wordcloud_aspect(data, fig, ax, bank)

#         else:
#             simple_wordcloud_aspect(data_competitors[i-1], fig, ax, competitors[i-1])
    
#     st.pyplot(fig)
    
    
    # ЕЩЕ ПО МНЕНИЯМ
    
    
    
    
    
    
    
    
    