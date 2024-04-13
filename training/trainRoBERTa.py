from RoBERTamodel import BERTmodel
import pandas as pd
import os
import argparse

print("Checkpoint 0")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset used')
    parser.add_argument('--max_epochs', type=int, help='Maximum number of epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--num_gpus', type=int, help='Number of GPUs to use')
    parser.add_argument('--data_dir', type=str, help='Dataset directory used')
    parser.add_argument('--train_path', type=str, help='Training set directory')
    parser.add_argument('--val_path', type=str, help='Validation set directory')
    parser.add_argument('--test_path', type=str, help='Testing set directory') 
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--num_classes', type=int, help='Number of classes')
    args = parser.parse_args()
    return args

def main(args):
    
    model = BERTmodel()

    model.from_pretrained(model_name="roberta-base")

    print("Training starting...")

    CACHED_FPATH = os.path.join("/scratch/ndekhil/cmdpred2/NRV/ROBERTA/checkpoints/", f"epochs_{args.max_epochs}_batch_{args.batch_size}_lr_{args.lr}")

    os.makedirs(CACHED_FPATH, exist_ok=True)

    model.train(
            data_dir = args.data_dir,
            model_name = "roberta-base",
            max_length = 512,
            train_path = args.train_path,
            test_path = args.test_path,
            val_path = args.val_path,
            batch_size = args.batch_size,
            lr = args.lr,
            num_classes = args.num_classes,
            n_training_steps = 320,
            deterministic = True,
            max_epochs = args.max_epochs,
            num_gpus = args.num_gpus,
            outputdir=CACHED_FPATH
            )
if __name__ == "__main__":
    args = parse_args()
    main(args)
