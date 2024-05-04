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


from test_script import get_preds
# from data_and_models import check_data, get_pred, convert_df, plot_shap_explanation
import numpy as np
import matplotlib.pyplot as plt
# from aux_functions import *
import os
import re
from aux_functions import *
from collections import Counter


def process_punctuation_txt(review):
  # Обрабатываем знаки препинания
    review = re.sub(rf'(\W)', rf' \1 ', review[:-14])

    review = re.sub(rf'\s+', rf' ', review)

    for punct in '.,':
        review = re.sub(rf'([\d\{punct}]) (\{punct}) ([\d\{punct}])', rf'\1\2\3', review)

    for punct in '!?':
        review = re.sub(rf'(\{punct}) (\{punct}) (\{punct})', rf'\1\2\3', review)

    if review[-1] == ' ':
        review = review[:-1]

    if review[-1] not in '.!?':
        review += ' .'
    review += '#### []'

    return review


def add_spaces(review):
    pattern=r'(\w\w[а-яё])([А-Я]\w\w)'
    review=re.sub(pattern, r'\1 \2', review)
    return review
    


def process_reviews(reviews):
    
    reviews = reviews.apply(lambda x: re.sub(r'\s+', ' ', x).strip())
    # reviews = reviews.apply(lambda x: x + '#### #### ####[]')
    reviews = reviews.apply(add_spaces)
    reviews = reviews.apply(process_punctuation_txt)
        
    return reviews.values.tolist()



def page_analysis_from_file():
    ALLOWED_EXTENSIONS = set(['txt']) # TODO: add csv

    file = st.file_uploader("Загрузите файл с данными, которые хотите проанализировать",
                            type=ALLOWED_EXTENSIONS)
    if file:

        preds_hist = pd.read_csv('data/preds_hist.csv')

        if '.' in file.name and file.name.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
                st.success(f"Файл {file.name} успешно загружен!")

                df = file_to_triplets(file)
        # print(preds)    
        
        if df is not None:

            triplets = df['pred_text'].values.tolist()
            all_triplets = []
            for label in triplets:
                # all_triplets.extend(eval(label))
                all_triplets.extend(label)

            data = []
            for triplet in all_triplets:
                at, ot, sp = triplet
                at = lemmatize(at.lower())
                ot = lemmatize(ot.lower())

                data.append([at, ot, sp, (at, ot, sp), (at, ot)])
            data = pd.DataFrame(data)
            data.columns = ['aspect', 'opinion', 'sentiment', 'triplet', 'aspect_opinion']
            data = data[~data.aspect.apply(is_stop_word)]
            st.write(data)
            
            analyse_file(df, data, all_triplets)

            
            
            
            
            


    # corr_y1 = corr1['y'].dropna()
    # corr_y1 = corr_y1.sort_values(key=lambda x: abs(x))[:-1]
    # plt.barh(corr_y1.index, abs(corr_y1))
    # plt.title('Корреляция с целевой переменной', fontsize=25, fontweight='bold')
    # ax.bar_label(ax.containers[0], labels=corr_y1.apply(lambda x: round(x, 3)), fmt='%.3f')
    # st.pyplot(fig)

                
                
      
def file_to_triplets(file):
     if file:
    
        if file.name.rsplit('.', 1)[1].lower() == 'txt':
            # with open(file) as f:
            lines = [line.decode()[:-1] for line in file.readlines()]
            if '####' in lines[0]: 

                # if st.radio(
                #         'В ваших данных присутсвуют триплеты. Хотите проверить точность на них?',
                #         ['Удалить', 'Оценить точность']) == 'Оценить точность':
                #     has_target = True
                # else:
                has_target = False
                lines = [line.rsplit('####')[0] for line in lines]
            
            
            lines = process_reviews(pd.Series(lines))
                    
            

        with open('data/uploaded_file.txt', 'w') as f:
            for line in lines:
                f.write(line+'\n')
                # st.write(line[-1])



        pred_button = st.button("Начать")
        # st.caption('Если Вы не нажали кнопку "Начать", снизу отображаются предсказания для вашего предыдущего файла')
        if pred_button:
            if file is not None:
                file.seek(0)
                st.write('Started inference')
                preds = get_preds(lines, model_path='model_58.pt', test_path='data/uploaded_file.txt')
                st.write(preds['pred_text'])
                
            return preds
    
    
    
    
def analyse_file(df, data, all_triplets):
    
    
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
    
    # df.to_csv('aaa.csv')
    
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
        if len(x) >= 20 or len(df) < 30:
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
        if len(x) >= 20 or len(df) <= 30:
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
