import requests
import pandas as pd
import pickle
from tqdm import tqdm
from analysis_from_file import process_reviews
from test_script import get_preds
import streamlit as st
from aux_functions import *

cats = ["serviceLevel", "mortgage", "credits", "creditCards", "debitCards", "savings", "remoteService", "other"]



with open('data/constants/max_dates.pickle', 'rb') as f:
    max_dates = pickle.load(f)
    

    
def aste_inference(reviews):
        with open('data/uploaded_file.txt', 'w') as f:
            for line in reviews:
                f.write(line+'\n')




        # st.write('Started inference')
        preds = get_preds(reviews, model_path='model_58.pt', test_path='data/uploaded_file.txt', type='inference')
        # st.write(preds['pred_text'])
                
    
        return preds


def parse_one_bank(bank):
    # st.write(bank, max_dates, bank in max_dates)

    res = []
    for cat in cats:
        res_cat = []
        f = 1
        for i in tqdm(range(0, 1001)):
            json = requests.get(f"https://www.sravni.ru/proxy-reviews/reviews/?filterBy=withRates&fingerPrint=75da44baf111f15bf7e15eb1a990450d&isClient=false&locationRoute=&newIds=true&orderBy=byDate&pageIndex={i}&pageSize=10&reviewObjectId=5bb4f769245bc22a520a6353&reviewObjectType=banks&specificProductId=&tag={cat}&withVotes=true").json()
            for item in json["items"]:
                # st.write(0)
                if cat in max_dates[bank] and max_dates[bank][cat] >= item['date']:
                    # st.write(max_dates[bank][cat], item['date'])
                    f = 0
                    break
            if f == 0: break

            res_cat.append(( item["date"],item["text"],item["title"], item["problemSolutionDate"], item["rating"], cat))
            if len(json["items"]) == 0: break
        res.extend(res_cat)

    final = pd.DataFrame(res, columns=["date", "text", "title", "solution_date", "rating", "category"])
    current = pd.read_parquet(f'data/results_sravni_{bank}.parquet')


    reviews = final['text']
    reviews = process_reviews(reviews)
    
    st.write(f'Банк: {map_bank_reverse[bank]}, Нашлось {len(reviews)} новых отзывов')
    
    # st.write(2)
    if len(reviews) == 0: 
        return 0

    results = aste_inference(reviews)
    results['time'] = final['date']
    results['rating'] = final['rating']
    # st.write(3)
    results['old_text'] = final['text']
    results['category'] = final['category']
    results['pred_text'] = results['pred_text'].apply(lambda x: str(x))
    results['pred'] = results['pred'].apply(lambda x: str(x))
    # results.to_parquet(f'drive/MyDrive/sirius_ai/results_sravni_{bank}.parquet', index=False)
    # print(preds)
    results = results[['text', 'pred', 'pred_text', 'time', 'rating', 'old_text', 'category']]
    current = pd.concat([results, current], axis=0, ignore_index=False)
    
    # st.write(3)
    
    current.to_parquet(f'data/results_sravni_{bank}.parquet')
    
    max_dates_parsed = results.groupby('category')['time'].agg(max).to_dict()
    for cat in max_dates_parsed.keys():
        if cat in max_dates[bank]:
            max_dates[bank][cat] = max(max_dates[bank][cat], max_dates_parsed[cat])
        else:
            max_dates[bank][cat] = max_dates_parsed[cat]
            
    
    with open('data/constants/max_dates.pickle', 'wb') as f:
         pickle.dump(max_dates, f)



def parse_data():
    st.title('Парсинг отзывов')
    pred_button = st.button("Начать")
    
    if pred_button:
        # st.write(1)
        parse_one_bank('tink')
        # st.write(1)
        parse_one_bank('sber')
        # st.write(1)
        parse_one_bank('alpha')
        # st.write(1)
        parse_one_bank('vtb')
    
   
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    with open('data/constants/max_dates.pickle', 'rb') as f:
        max_dates = pickle.load(f)
    
    
    
    # Обновляем max_date
    