import torch
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import faiss
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "DeepMount00/Mistral-RAG"

def recomendation(MODEL_NAME, category):
  model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).eval().to(device)
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  
  def get_embeddings(text):
      """ Генерация эмбеддингов для текста """
      input_ids = tokenizer(text, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
      with torch.no_grad():
          outputs = model(input_ids=input_ids, output_hidden_states=True)
      hidden_states = outputs.hidden_states[-1]
      embeddings = torch.mean(hidden_states, dim=1).squeeze()
      return embeddings.float().cpu().numpy()  


  data = pq.read_table('results_sravni_tink.parquet').to_pandas()

  d = embeddings.shape[1]
  index = faiss.IndexFlatL2(d)
  index.add(embeddings)
  D, I = index.search(embeddings, 30) 

  closest_texts = filtered_data.iloc[I.flatten()]

  improvement_prompts = f"""
  Это отзывы на банк. Найди из них 30 релевантных в категории "{category}". Составь список из того что нужно сделать банку для улучшения пользовательского опыта. Список должен содержать три пункта.
  """
  model_inputs = tokenizer(improvement_prompts, return_tensors="pt").input_ids.to(device)
  with torch.no_grad():
      generated_ids = model.generate(model_inputs, max_length=512, temperature=0.001, num_return_sequences=1)
  improvements = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

  return improvements
