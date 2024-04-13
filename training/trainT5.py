import os
import time
import datetime
import torch
import pandas as pd
import argparse

from T5model import SimpleT5
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', type=int, help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--dataloader_num_workers', type=int, help='Number of dataloader workers')
    parser.add_argument('--num_gpus', type=int, help='Number of GPUs to use')
    args = parser.parse_args()
    return args

def drop_NaN(df):
  non_str_index = df[df['context'].apply(lambda x: not isinstance(x, str))].index

  if not non_str_index.empty:
      print(f"Dropping row at index {non_str_index[0]} because it contains a non-str value in 'context' column.")
      df.drop(index=non_str_index[0], inplace=True)
  else:
      print("All rows in the 'context' column are of type str.")

  non_str_index = df[df['class'].apply(lambda x: not isinstance(x, str))].index

  if not non_str_index.empty:
      print(f"Dropping row at index {non_str_index[0]} because it contains a non-str value in 'class' column.")
      df.drop(index=non_str_index[0], inplace=True)
  else:
      print("All rows in the 'class' column are of type str.")
  return(df)


def main(args):

    train_df = pd.read_csv("/scratch/ndekhil/cmdpred2/NRV/T5/dataset/training_90.csv")
    test_df = pd.read_csv("/scratch/ndekhil/cmdpred2/NRV/T5/dataset/testing_10.csv")


    train_df = drop_NaN(train_df)
    test_df = drop_NaN(test_df)
    
    max_src_tok_len = max([len(x.split()) for x in train_df['context']]) + 10
    print("Max source toklength={}".format(max_src_tok_len))
    
    model = SimpleT5(source_max_token_len=max_src_tok_len, target_max_token_len=10)
    model.from_pretrained("t5", "t5-base")


    CACHED_FPATH = os.path.join("/scratch/ndekhil/cmdpred2/NRV/T5/models/", f"epochs_{args.max_epochs}_batch_{args.batch_size}_gpus_{args.num_gpus}")

    os.makedirs(CACHED_FPATH, exist_ok=True)  


    model.train(
        train_df=train_df,
        eval_df=test_df,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        outputdir=CACHED_FPATH,
        save_only_last_epoch=False,
        num_gpus=args.num_gpus
    )

if __name__ == "__main__":
    args = parse_args()
    main(args)