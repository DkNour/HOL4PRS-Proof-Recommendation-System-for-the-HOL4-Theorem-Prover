import pandas as pd
from tqdm import tqdm
import torch
import json
from T5model import SimpleT5
from transformers import AutoTokenizer
from RoBERTamodel import BertTextClassifier

def predict(text, model, tokenizer, max_length=512, top_k=7):
  encoding = tokenizer.encode_plus(
    text,
    max_length=max_length,
    return_token_type_ids=False,
    padding="max_length",
    truncation=True,
    return_attention_mask=True,
    return_tensors='pt',
  )
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  encoding["input_ids"], encoding["attention_mask"] = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)
  _, test_prediction = model(encoding["input_ids"], encoding["attention_mask"])
  top_k_values, top_k_indices = torch.topk(test_prediction, k=top_k, dim=-1)
  with open("labelencoder.json", 'r') as file:
      data = json.load(file)
  result = {}
  preds = top_k_indices.tolist()[0]
  for key, value in data.items():
      if value in preds:
          result[value] = key

  preds = list(result.values())

  return preds  

def main():


    sequence = input("Proof state: ")

    t5_model = SimpleT5(source_max_token_len=512, target_max_token_len=10)
    t5_model.from_pretrained("t5", "t5-base")
    t5_model.load_model(model_dir=best_mobels/fullt5, use_gpu=True)
    t5_recom = t5_model.predict(sequence, num_return_sequences=7, num_beams=7)[:7]

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    roberta_model = BertTextClassifier.load_from_checkpoint(checkpoint_path=best_mobels/fullroberta bert_model="roberta-base", n_classes=162)
    roberta_recom = predict(sequence, roberta_model, tokenizer)

    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    bert_model = BertTextClassifier.load_from_checkpoint(checkpoint_path=best_mobels/fullbert, bert_model="bert-base-cased", n_classes=162)
    bert_recom = predict(sequence, bert_model, tokenizer)

    print("T5 reccommendations are: ", t5_recom)
    print("BERT reccommendations are: ", bert_recom)
    print("RoBERTa reccommendations are: ", roberta_recom)



if __name__ == "__main__":
    main()