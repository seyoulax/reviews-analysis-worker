#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# ====================================
# @Project ：insights-span-aste
# @IDE     ：PyCharm
# @Author  ：Hao,Wireless Zhiheng
# @Email   ：hpuhzh@outlook.com
# @Date    ：05/08/2022 9:57 
# ====================================
import os

import argparse
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from models.model import SpanAsteModel
from utils.tager import SpanLabel, RelationLabel
import time
import pickle


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="banki", type=str)
    parser.add_argument(
        "--bert_model",
        default="ai-forever/ruBert-base",
        type=str,
        help="Path to pre-trained Bert model or shortcut name",
    )

    parser.add_argument(
        "--model_path",
        default="checkpoint/model_best/model.pt",
        type=str,
        help="Path to checkpoint",
    )

    parser.add_argument(
        "--test_path",
        default="data/banki/test_triplets.txt",
        type=str,
        help="Path to data",
    )

    parser.add_argument("--max_seq_length", default=512, type=int)

    parser.add_argument(
        "--type", default='test', choices=['test', 'inference'], help="Проводится оценка на тестовом датасете или инференс"
    )

    parser.add_argument(
        "--verbose", default='False', choices=['False', 'True']
    )

    parser.add_argument("--use_additional_head", default='False', choices=['False', 'True'], help="Use additional token classification head or not")
    parser.add_argument("--use_neutral_class", default='False', choices=['False', 'True'], help="Will there be a neutral class in the data")
    parser.add_argument("--span_maximum_length", default=5, type=int, help="The maximum length of span to be considered as an aspect/opinion")
    parser.add_argument("--load_args", type=str, default='True', help="Загрузить параметры из файла или нет")

    args = parser.parse_args()

    model_path = args.model_path
    test_path = args.test_path
    verbose = args.verbose
    args_type = args.type

    args.use_additional_head = {'False': False, 'True': True}[args.use_additional_head]
    args.use_neutral_class = {'False': False, 'True': True}[args.use_neutral_class]
    
    if args.load_args == 'True':
        with open('checkpoint/args.pkl', 'rb') as f:
            args = pickle.load(f)
    
        args.model_path = model_path
        args.test_path = test_path
        args.verbose = verbose
        args.type = args_type

    return args


def compute_f1(labels, preds):
  n_tp, n_pred, n_gold = 0, 0, 0
  labels = labels.values
  preds = preds.values

  for label, pred in zip(labels, preds):
    n_pred += len(pred)
    n_gold += len(label)

    for triplet in label:
      if triplet in pred:
        n_tp += 1

  # print(n_tp, n_pred, n_gold)

  precision = float(n_tp) / float(n_pred) if n_pred != 0 else 0
  recall = float(n_tp) / float(n_gold) if n_gold != 0 else 0
  f1 = (
      2 * precision * recall / (precision + recall)
      if precision != 0 or recall != 0
      else 0
  )
  scores = {"precision": precision, "recall": recall, "f1": f1}

  return scores
  



args = init_args()


device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print(f"using device:{device}")
# tokenizer
tokenizer = BertTokenizer.from_pretrained(args.bert_model)

print("Building SPAN-ASTE model...")
# get dimension of target and relation
target_dim, relation_dim = len(SpanLabel), len(RelationLabel)

if args.use_neutral_class == False:
    relation_dim -= 1

# build span-aste model
model = SpanAsteModel(
    args.bert_model,
    target_dim,
    relation_dim,
    device=device,
    use_additional_head=args.use_additional_head,
    span_maximum_length=args.span_maximum_length
)

model.load_state_dict(torch.load(args.model_path, map_location=torch.device(device)))
model.to(device)
model.eval()

with open(args.test_path, "r", encoding="utf8") as f:
    data = f.readlines()
res = []
# sliced_review_idx_to_review_idx = []

for line_idx, d in tqdm(enumerate(data)):
    text, label = d.strip().split("####")
    words = text.split()

    input_ids = [101]

    word_pos_to_token_pos = dict()
    
    count_ids = 1

    for i, word in enumerate(words): # Идем по словам, каждое переводим в индексы токенов, добавляем в общий список токенов
        ids_word  = tokenizer.encode(word, add_special_tokens=False)
        input_ids.extend(ids_word)

        word_pos_to_token_pos[i] = (count_ids, count_ids + len(ids_word))
        count_ids += len(ids_word)

        # if count_ids >= args.max_seq_len - 1: # Вышли за ограничение по токенам. -1 ставим для end token (102).

        #     if word_pos_to_token_pos[i][1] > args.max_seq_len - 1: # Если токены из последнего токена выходят за ограничение, последнее слово удаляем
        #         del word_pos_to_token_pos[i]
            
        #     break

    count_parts = 1 # На сколько частей нарезаем отзыв
    word_idx = 0

    predict = []
    predict_text = []

    for i, word in enumerate(words):
        if i==len(words)-1 or word_pos_to_token_pos[i+1][1] > (args.max_seq_len - 2) * count_parts:
            if i==len(words)-1:
                input_ids_part = input_ids[word_pos_to_token_pos[word_idx][0]:]
                # print('b', len(input_ids_part))
            else:
                input_ids_part = input_ids[word_pos_to_token_pos[word_idx][0]:word_pos_to_token_pos[i][1]]
                # print('c', len(input_ids_part))
            
            
            # if count_parts > 1:
            input_ids_part = [101] + input_ids_part

            word_pos_to_token_pos_part = {num: word_pos_to_token_pos[j] for num, j in enumerate(range(word_idx, i+1))}
            word_pos_to_token_pos_part = {num: (word_pos_to_token_pos_part[num][0] + 1 - word_pos_to_token_pos[word_idx][0],
                word_pos_to_token_pos_part[num][1] + 1 - word_pos_to_token_pos[word_idx][0]) for num in word_pos_to_token_pos_part}

            word_ids = list(range(word_idx, i+1))

            if len(input_ids_part) <= 2:
                continue

            input_ids_part.append(102)

            count_parts += 1

            # sliced_review_idx_to_review_idx.append(line_idx)


    
            attention_mask = torch.tensor([1] * len(input_ids_part), device=device)
            length = len(input_ids_part)

            # print(tokenizer.decode(input_ids_part))
            # print(word_pos_to_token_pos_part)
            # print(len(tokenizer.decode(input_ids_part).split()))
            # forward
            bio_probability, spans_probability, span_indices, relations_probability, candidate_indices = model(
                        torch.tensor(input_ids_part, device=device).reshape(1, -1), attention_mask.reshape(1, -1), 
                        length, (word_pos_to_token_pos_part, ))


            candidate_indices = candidate_indices[0]
            relations_probability = relations_probability.squeeze(0)

            predicted_labels = relations_probability.argmax(-1)
            predicted_rels = predicted_labels.nonzero().squeeze(1).cpu().numpy()

            # if predicted_rels.shape != torch.Size([]):
            # print(predicted_rels)
            # if isinstance(predicted_rels, int):
            #     predicted_rels = [predicted_rels]
            #     print('new', predicted_rels)
            for idx in predicted_rels:
                can = candidate_indices[idx]
                
                sentiment = RelationLabel(predicted_labels[idx].item()).name

                a, b, c, d = can
                
                # print(can, word_pos_to_token_pos[word_idx][0] - 1)

                a += word_idx# - 1 
                b += word_idx# - 1
                c += word_idx# - 1
                d += word_idx# - 1


                aspect = ' '.join(words[a:b])
                opinion = ' '.join(words[c:d])

                predict_text.append((aspect, opinion, sentiment))
                predict.append((list(range(a, b)), list(range(c, d)), sentiment))

            # else:
            #     print('AAAAAAAAA', predicted_rels)
            
            word_idx = i + 1


    if args.verbose == 'True':
      print("text:", text)
      print("predict", predict_text)
      print("pred_ids", predict)


    if args.type == 'test':
      labels = []
      labels_text = []

      for l in eval(label):
          a, o, sm = l
          labels.append((a, o, sm))
          a = " ".join([words[i] for i in a])
          o = " ".join([words[i] for i in o])
          labels_text.append((a, o, sm))

      if args.verbose == 'True':
        print("label", labels_text)
        print("label_ids", labels)
      res.append({"text": text, "pred": predict, "label": labels, "pred_text": predict_text, "label_text": labels_text})
    
    else:
      res.append({"text": text, "pred": predict, "pred_text": predict_text})

df = pd.DataFrame(res)
df.to_csv("results.csv", index=False)

if args.type == 'test':
  print(compute_f1(df['label'], df['pred']))