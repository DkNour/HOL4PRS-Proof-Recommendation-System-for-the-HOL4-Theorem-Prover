import pandas as pd
from transformers import MarianMTModel, MarianTokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset
from sklearn.model_selection import train_test_split
import torch
import editdistance
from nltk.metrics import edit_distance
from Levenshtein import distance as levenshtein_distance
import argparse
import os

# Initialize the argument parser
parser = argparse.ArgumentParser(description="Fine-tune a transformer model for proof translation")

# Add arguments for MODEL_CHECKPOINT, BATCH_SIZE, FILENAME, TEST_FILE, learning_rate, weight_decay, num_train_epochs
parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training and evaluation")
parser.add_argument('--filename', type=str, default="train.xlsx", help="Training file name")
parser.add_argument('--test_file', type=str, default="test.xlsx", help="Test file name")
parser.add_argument('--learning_rate', type=float, default=3e-4, help="Learning rate for training")
parser.add_argument('--weight_decay', type=float, default=0.02, help="Weight decay for training")
parser.add_argument('--num_train_epochs', type=int, default=1, help="Number of training epochs")

# Parse the arguments
args = parser.parse_args()

# Load your data
train_data_path = args.filename
test_data_path = args.test_file

data = pd.read_excel(train_data_path)
test = pd.read_excel(test_data_path)

FILENAME = args.filename
NAME = name = os.path.splitext(FILENAME)[0]

#data = data.drop(columns=['Num_Tactics'])

# Ensure all data is string
data['Theorem'] = data['Theorem'].astype(str)
data['Proof'] = data['Proof'].astype(str)

# Ensure all data is string
test['Theorem'] = test['Theorem'].astype(str)
test['Proof'] = test['Proof'].astype(str)

# Drop rows where the 'Proof' column is NaN
test = test.dropna(subset=['Proof'])
test = test.dropna(subset=['Theorem'])
data = data.dropna(subset=['Proof'])
data = data.dropna(subset=['Theorem'])

# Prepare the dataset
#train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
#train_dataset = Dataset.from_pandas(train_data)
#test_dataset = Dataset.from_pandas(test_data)

train_dataset = Dataset.from_pandas(data)
test_dataset = Dataset.from_pandas(test)

model_name = '/lustre04/scratch/ndekhil/env/MarianMT/Dataset4/Helsinki-NLP/opus-mt-en-ROMANCE'
tokenizer = MarianTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    tokenized_inputs = tokenizer(examples['Theorem'], max_length=512, truncation=True, padding="max_length")
    tokenized_targets = tokenizer(examples['Proof'], max_length=512, truncation=True, padding="max_length")
    return {
        "input_ids": tokenized_inputs['input_ids'],
        "attention_mask": tokenized_inputs['attention_mask'],
        "labels": tokenized_targets['input_ids']
    }

train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# Print a sample to check
print(train_dataset[0])

model = MarianMTModel.from_pretrained(model_name)

training_args = Seq2SeqTrainingArguments(
    output_dir=f"./{NAME}_results-lr{args.learning_rate}-batch{args.batch_size}-epoch{args.num_train_epochs}-decay{args.weight_decay}",
    evaluation_strategy='epoch',
    learning_rate=args.learning_rate,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    weight_decay=args.weight_decay,
    save_total_limit=3,
    num_train_epochs=args.num_train_epochs,
    predict_with_generate=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

trainer.train()

trainer.save_model(f"{NAME}_FineTunedTMarianMT-lr{args.learning_rate}-batch{args.batch_size}-epoch{args.num_train_epochs}-decay{args.weight_decay}")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Assuming you have a trained model and test_dataset ready

# Generate predictions on the test dataset
test_results = trainer.predict(test_dataset)
predictions = test_results.predictions

# Decode predictions
decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)

# Prepare data for saving to CSV
# Extract test inputs and labels (as text) for comparison
test_inputs = tokenizer.batch_decode(test_dataset['input_ids'], skip_special_tokens=True)
test_labels = tokenizer.batch_decode(test_dataset['labels'], skip_special_tokens=True)

# Create a DataFrame
output_df = pd.DataFrame({
    'Theorem': test_inputs,
    'Expected Proof': test_labels,
    'Predicted Proof': decoded_predictions
})

# Save DataFrame to CSV
output_df.to_csv(f'{NAME}_model_predictions-lr{args.learning_rate}-batch{args.batch_size}-epoch{args.num_train_epochs}-decay{args.weight_decay}.csv', index=False)

print("_Predictions saved to model_predictions.csv")

#Evaluation
# Load your CSV file
df = pd.read_csv(f'{NAME}_model_predictions-lr{args.learning_rate}-batch{args.batch_size}-epoch{args.num_train_epochs}-decay{args.weight_decay}.csv')

# Function to calculate edit distance
def calculate_edit_distance(true_seq, pred_seq):
    # Convert both sequences to strings and handle NaN values
    true_seq = str(true_seq) if pd.notna(true_seq) else ""
    pred_seq = str(pred_seq) if pd.notna(pred_seq) else ""
    return editdistance.eval(true_seq.split(), pred_seq.split())

# Apply the function to each row in the DataFrame
df['edit_distance'] = df.apply(lambda row: calculate_edit_distance(row['Expected Proof'], row['Predicted Proof']), axis=1)

# Calculate normalized edit distance
#df['normalized_edit_distance'] = df.apply(lambda row: row['edit_distance'] / len(row['y_true_eng'].split()), axis=1)
#capping at 1
df['normalized_edit_distance'] = df.apply(lambda row: min(row['edit_distance'] / len(row['Expected Proof'].split()), 1), axis=1)


# Display the DataFrame with the new columns
print(df.head())

# Optionally, save the updated DataFrame back to a CSV
df.to_csv(f"{NAME}_Normalized scores saved to normalized_scores-lr{args.learning_rate}-batch{args.batch_size}-epoch{args.num_train_epochs}-decay{args.weight_decay}.csv", index=False)

average_normalized_edit_distance = df['normalized_edit_distance'].mean()

# Display the averages
print(f"Average Edit Distance: {average_normalized_edit_distance}")

# Load the file
file_path = f"{NAME}_model_predictions-lr{args.learning_rate}-batch{args.batch_size}-epoch{args.num_train_epochs}-decay{args.weight_decay}.csv"
df = pd.read_csv(f"{NAME}_model_predictions-lr{args.learning_rate}-batch{args.batch_size}-epoch{args.num_train_epochs}-decay{args.weight_decay}.csv")

def jaccard_similarity(list1, list2):
    """Calculate Jaccard Similarity between two lists of words."""
    set1 = set(list1.split())
    set2 = set(list2.split())

    intersection = set1.intersection(set2)
    union = set1.union(set2)

    if len(union) == 0:  # To avoid division by zero
        return 0

    return len(intersection) / len(union)

# Apply the Jaccard similarity calculation to each row
df['jaccard_similarity'] = df.apply(
    lambda row: jaccard_similarity(str(row['Expected Proof']), str(row['Predicted Proof'])), axis=1
)

# Save the updated dataframe to a new CSV (optional)
df.to_csv(f"{NAME}__jaccard_similarity_output-lr{args.learning_rate}-batch{args.batch_size}-epoch{args.num_train_epochs}-decay{args.weight_decay}.csv", index=False)

# Show the updated dataframe
print(df[['Expected Proof', 'Predicted Proof', 'jaccard_similarity']].head())

# Compute the aggregated similarity (mean Jaccard similarity for all rows)
aggregated_similarity = df['jaccard_similarity'].mean()

# Print the aggregated similarity
print(f"Aggregated Jaccard Similarity: {aggregated_similarity}")

#ft_model_tokenizer = T5Tokenizer.from_pretrained("FineTunedTransformer")
#ft_model = T5ForConditionalGeneration.from_pretrained("FineTunedTransformer")

# Load the file
file_path = f"{NAME}__jaccard_similarity_output-lr{args.learning_rate}-batch{args.batch_size}-epoch{args.num_train_epochs}-decay{args.weight_decay}.csv"
df = pd.read_csv(file_path)

# Function to calculate Levenshtein Similarity Percentage
def levenshtein_similarity_percentage(seq1, seq2):
    """Calculate Levenshtein similarity percentage between two sequences."""
    lev_distance = levenshtein_distance(seq1, seq2)
    max_length = max(len(seq1), len(seq2))

    # If max_length is zero (both sequences are empty), return 100% similarity
    if max_length == 0:
        return 100.0

    similarity_percentage = (1 - lev_distance / max_length) * 100
    return similarity_percentage

# Apply the Levenshtein similarity percentage calculation to each row
df['levenshtein_similarity_percentage'] = df.apply(
    lambda row: levenshtein_similarity_percentage(str(row['Expected Proof']), str(row['Predicted Proof'])), axis=1
)

# Save the updated dataframe to a new CSV file
output_file_path = f"{NAME}__jaccard_similarity_with_levenshtein_percentage-lr{args.learning_rate}-batch{args.batch_size}-epoch{args.num_train_epochs}-decay{args.weight_decay}.csv"
df.to_csv(output_file_path, index=False)

# Display the first few rows of the updated dataframe
print(df[['Expected Proof', 'Predicted Proof', 'levenshtein_similarity_percentage']].head())


# Function to calculate word-level Levenshtein similarity percentage
def levenshtein_similarity_percentage_word_level(seq1, seq2):
    """Calculate Levenshtein similarity percentage between two sequences of words."""
    # Split sequences into words (tactics)
    words1 = seq1.split()
    words2 = seq2.split()

    # Calculate Levenshtein distance based on words (not characters)
    lev_distance = edit_distance(words1, words2)
    max_length = max(len(words1), len(words2))

    # If max_length is zero (both sequences are empty), return 100% similarity
    if max_length == 0:
        return 100.0

    similarity_percentage = (1 - lev_distance / max_length) * 100
    return similarity_percentage

# Load the file
file_path = f"{NAME}__jaccard_similarity_output-lr{args.learning_rate}-batch{args.batch_size}-epoch{args.num_train_epochs}-decay{args.weight_decay}.csv"
df = pd.read_csv(file_path)

# Apply the word-level Levenshtein similarity percentage calculation to each row
df['levenshtein_similarity_percentage_word_level'] = df.apply(
    lambda row: levenshtein_similarity_percentage_word_level(str(row['Expected Proof']), str(row['Predicted Proof'])), axis=1
)

# Save the updated dataframe to a new CSV file
output_file_path = f"{NAME}__Jaccard Similarity with Levenshtein Word Level results saved to jaccard_similarity_with_levenshtein_word_level-lr{args.learning_rate}-batch{args.batch_size}-epoch{args.num_train_epochs}-decay{args.weight_decay}.csv"
df.to_csv(output_file_path, index=False)

# Display the first few rows of the updated dataframe
print(df[['Expected Proof', 'Predicted Proof', 'levenshtein_similarity_percentage_word_level']].head())

# Calculate the aggregated similarity percentage (average of all rows)
aggregated_similarity_percentage = df['levenshtein_similarity_percentage_word_level'].mean()

# Print the aggregated similarity percentage
print(f"Aggregated Word-Level Levenshtein Similarity Percentage: {aggregated_similarity_percentage:.2f}%")