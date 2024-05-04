import streamlit as st
import pandas as pd
import numpy as np
import torch



def inference(review, checkpoint_path):
    answer = 'ААААА ББББ ВВВВ'
    return answer



def generate_answer():
    review = st.text_area("Введите отзыв", 'В вашем банке меня обманул сотрудник, условия мелким шрифтом прописаны, ничего не понятно')
    checkpoint_path = 'aaa.pt'


    answer = inference(review, checkpoint_path)

    st.subheader('Ответ:')
    st.write(answer)
