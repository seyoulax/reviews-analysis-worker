import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig



def autoanswer(feedback, path):
  base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
  bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_use_double_quant=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16
  )

  base_model = AutoModelForCausalLM.from_pretrained(
      base_model_id,  # Mistral, same as before
      quantization_config=bnb_config,  # Same quantization config as before
      device_map="auto",
      trust_remote_code=True,
  )

  eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)



  from peft import PeftModel

  ft_model = PeftModel.from_pretrained(base_model, path)



  eval_prompt = f" есть отзыв клиента банка: ('{feedback}') Напиши на него ответ от компании: "

  model_input = eval_tokenizer(eval_prompt, return_tensors="pt").to("cuda")

  ft_model.eval()
  with torch.no_grad():
      generated_tokens = ft_model.generate(**model_input, max_new_tokens=150, repetition_penalty=1.15)[0]
      decoded_output = eval_tokenizer.decode(generated_tokens, skip_special_tokens=True)
      prompt_length = len(eval_prompt)    
      print(decoded_output[prompt_length:])
