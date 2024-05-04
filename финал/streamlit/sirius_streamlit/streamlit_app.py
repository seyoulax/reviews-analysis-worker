# from data_and_models import *
from analysis_from_file import page_analysis_from_file
from analysis_hist import analyze_hist_data
from competitors_analysis import analyze_competitors
from parse_reviews import parse_data
from answering_machine import generate_answer
# from preds_by_hand import page_prediction
# from cats_info import get_similar_categories
# from business import page_business_info
import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
# from matplotlib.ticker import FuncFormatter
# import matplotlib.ticker as ticker
# from data_and_models import check_data

def main():
    # st.write(11111)
    st.sidebar.title("Навигация")
    pages = ['Анализ исторических данных', 'Анализ конкурентов', 'Предсказание и анализ для данных из файла',
            'Парсинг данных', 'Автоответчик']
    selected_page = st.sidebar.selectbox(
        'Доступные страницы',
        (pages),
        index=1)

    if selected_page == pages[0]:
        page_analysis_from_file()
    elif selected_page == pages[1]:
        analyze_hist_data()
    elif selected_page == pages[2]:
        analyze_competitors()
    elif selected_page == pages[3]:
        parse_data()
    elif selected_page == pages[4]:
        generate_answer()
    
if __name__ == '__main__':
    main()