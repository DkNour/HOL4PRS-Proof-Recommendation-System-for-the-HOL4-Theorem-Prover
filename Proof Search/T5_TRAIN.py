import argparse
import numpy as np
import pandas as pd
import editdistance
from nltk.metrics import edit_distance
import warnings
import torch
import transformers
from datasets import Dataset
from datasets import load_metric
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from Levenshtein import distance as levenshtein_distance

# Initialize the argument parser
parser = argparse.ArgumentParser(description="Fine-tune a transformer model for proof translation")

# Add arguments for MODEL_CHECKPOINT, BATCH_SIZE, FILENAME, TEST_FILE
parser.add_argument('--model_checkpoint', type=str, default="unicamp-dl_translation-pt-en-t5", help="Model checkpoint to use")
parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training and evaluation")
parser.add_argument('--filename', type=str, default="train.xlsx", help="Training file name")
parser.add_argument('--test_file', type=str, default="test.xlsx", help="Test file name")

# Parse the arguments
args = parser.parse_args()

# Assign parsed values to variables
MODEL_CHECKPOINT = args.model_checkpoint
BATCH_SIZE = args.batch_size
FILENAME = args.filename
TEST_FILE = args.test_file

BLEU = "bleu"
PROOF = "en"
PROOF_TEXT = "Proof"
EPOCH = "epoch"
INPUT_IDS = "input_ids"
GEN_LEN = "gen_len"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 512
MODEL_NAME = MODEL_CHECKPOINT.split("/")[-1]
LABELS = "labels"
PREFIX = ""
THEOREM = "pt"
THEOREM_TEXT = "Theorem"
SCORE = "score"
SOURCE_LANG = "pt"
TARGET_LANG = "en"
TRANSLATION = "translation"
UNNAMED_COL = "Unnamed: 0"

warnings.filterwarnings("ignore")

def postprocess_text(preds: list, labels: list) -> tuple:
    """Performs post processing on the prediction text and labels"""

    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def prep_data_for_model_fine_tuning(source_lang: list, target_lang: list) -> list:
    """Takes the input data lists and converts into translation list of dicts"""

    data_dict = dict()
    data_dict[TRANSLATION] = []

    for sr_text, tr_text in zip(source_lang, target_lang):
        temp_dict = dict()
        temp_dict[THEOREM] = sr_text
        temp_dict[PROOF] = tr_text

        data_dict[TRANSLATION].append(temp_dict)

    return data_dict


def generate_model_ready_dataset(dataset: list, source: str, target: str,
                                 model_checkpoint: str,
                                 tokenizer: AutoTokenizer):
    """Makes the data training ready for the model"""

    preped_data = []

    for row in dataset:
        inputs = PREFIX + row[source]
        targets = row[target]

        model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH,
                                 truncation=True, padding=True)

        model_inputs[TRANSLATION] = row

        # setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=MAX_INPUT_LENGTH,
                                 truncation=True, padding=True)
            model_inputs[LABELS] = labels[INPUT_IDS]

        preped_data.append(model_inputs)

    return preped_data



def compute_metrics(eval_preds: tuple) -> dict:
    """computes bleu score and other performance metrics """

    metric = load_metric("sacrebleu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {BLEU: result[SCORE]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]

    result[GEN_LEN] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}

    return result

translation_data = pd.read_excel(FILENAME)
#translation_data = translation_data.drop([UNNAMED_COL], axis=1)
#translation_data = translation_data.drop(columns=['Num_Tactics'])
translation_data = translation_data.dropna(subset=['Proof'])
translation_data = translation_data.dropna(subset=['Theorem'])
# If you want to reset the index after dropping rows
translation_data.reset_index(drop=True, inplace=True)

# Drop rows where the 'Proof' column is NaN
translation_data = translation_data.dropna(subset=['Proof'])

# Drop rows where the 'Proof' column is an empty string
translation_data = translation_data[translation_data['Proof'].str.strip() != '']

# If you want to reset the index after dropping rows
translation_data.reset_index(drop=True, inplace=True)

X = translation_data[THEOREM_TEXT]
y = translation_data[PROOF_TEXT]

x_train = X
y_train = y

test_translation_data = pd.read_excel(TEST_FILE)

#test_translation_data = test_translation_data.drop(columns=['Num_Tactics'])

# Drop rows where the 'Proof' column is NaN
test_translation_data = test_translation_data.dropna(subset=['Proof'])
test_translation_data = test_translation_data.dropna(subset=['Theorem'])

# Drop rows where the 'Proof' column is an empty string
test_translation_data = test_translation_data[test_translation_data['Proof'].str.strip() != '']

# If you want to reset the index after dropping rows
test_translation_data.reset_index(drop=True, inplace=True)

x_test = test_translation_data[THEOREM_TEXT]
y_test = test_translation_data[PROOF_TEXT]

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, shuffle=True, random_state=100)

print("FINAL X-TRAIN SHAPE: ", x_train.shape)
print("FINAL Y-TRAIN SHAPE: ", y_train.shape)
print("X-VAL SHAPE: ", x_val.shape)
print("Y-VAL SHAPE: ", y_val.shape)

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

training_data = prep_data_for_model_fine_tuning(x_train.values, y_train.values)

validation_data = prep_data_for_model_fine_tuning(x_val.values, y_val.values)

test_data = prep_data_for_model_fine_tuning(x_test.values, y_test.values)


print(list(training_data.keys()))

train_data = generate_model_ready_dataset(dataset=training_data[TRANSLATION], tokenizer=tokenizer, source=THEOREM, target=PROOF, model_checkpoint=MODEL_CHECKPOINT)

validation_data = generate_model_ready_dataset(dataset=validation_data[TRANSLATION], tokenizer=tokenizer, source=THEOREM, target=PROOF, model_checkpoint=MODEL_CHECKPOINT)

test_data = generate_model_ready_dataset(dataset=test_data[TRANSLATION], tokenizer=tokenizer, source=THEOREM, target=PROOF, model_checkpoint=MODEL_CHECKPOINT)

train_df = pd.DataFrame.from_records(train_data)

validation_df = pd.DataFrame.from_records(validation_data)

test_df = pd.DataFrame.from_records(test_data)

train_dataset = Dataset.from_pandas(train_df)

validation_dataset = Dataset.from_pandas(validation_df)

test_dataset = Dataset.from_pandas(test_df)

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)

model_args = Seq2SeqTrainingArguments( f"{MODEL_NAME}-finetuned-{SOURCE_LANG}-to-{TARGET_LANG}", evaluation_strategy=EPOCH, learning_rate=3e-4, per_device_train_batch_size=BATCH_SIZE, per_device_eval_batch_size=BATCH_SIZE, weight_decay=0.02, save_total_limit=3, num_train_epochs=1, predict_with_generate=True )

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer( model, model_args, train_dataset=train_dataset, eval_dataset=validation_dataset, data_collator=data_collator, tokenizer=tokenizer)

trainer.train()

trainer.save_model("FineTunedTransformer")

test_results = trainer.predict(test_dataset)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

predictions = []
test_input = test_dataset[TRANSLATION]

for input_text in tqdm(test_input):
    source_sentence = input_text[THEOREM]
    encoded_source = tokenizer(source_sentence,
                               return_tensors=THEOREM,
                               padding=True,
                               truncation=True)
    encoded_source.to(device) 
    translated = model.generate(**encoded_source, max_length=512)

    predictions.append([tokenizer.decode(t, skip_special_tokens=True) for t in translated][0])

y_true_en = []
y_true_pt = []

for input_text in tqdm(test_input):
    y_true_pt.append(input_text[THEOREM])
    y_true_en.append(input_text[PROOF])


output_df = pd.DataFrame({"y_true_port": y_true_pt, "y_true_eng": y_true_en, "predicted_text": predictions})


output_df.to_csv("output.csv", index=False)

# Load your CSV file
df = pd.read_csv('output.csv')

# Function to calculate edit distance
def calculate_edit_distance(true_seq, pred_seq):
    return editdistance.eval(true_seq.split(), pred_seq.split())

# Apply the function to each row in the DataFrame
df['edit_distance'] = df.apply(lambda row: calculate_edit_distance(row['y_true_eng'], row['predicted_text']), axis=1)

# Calculate normalized edit distance
#df['normalized_edit_distance'] = df.apply(lambda row: row['edit_distance'] / len(row['y_true_eng'].split()), axis=1)
#capping at 1
df['normalized_edit_distance'] = df.apply(lambda row: min(row['edit_distance'] / len(row['y_true_eng'].split()), 1), axis=1)


# Display the DataFrame with the new columns
print(df.head())

# Optionally, save the updated DataFrame back to a CSV
df.to_csv('path_to_your_file_with_scores_normalized.csv', index=False)

average_normalized_edit_distance = df['normalized_edit_distance'].mean()

# Display the averages
print(f"Average Edit Distance: {average_normalized_edit_distance}")

# Load the file
file_path = 'output.csv'
df = pd.read_csv(file_path)

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
    lambda row: jaccard_similarity(str(row['y_true_eng']), str(row['predicted_text'])), axis=1
)

# Save the updated dataframe to a new CSV (optional)
df.to_csv('jaccard_similarity_output.csv', index=False)

# Show the updated dataframe
print(df[['y_true_eng', 'predicted_text', 'jaccard_similarity']].head())

# Compute the aggregated similarity (mean Jaccard similarity for all rows)
aggregated_similarity = df['jaccard_similarity'].mean()

# Print the aggregated similarity
print(f"Aggregated Jaccard Similarity: {aggregated_similarity}")

#ft_model_tokenizer = T5Tokenizer.from_pretrained("FineTunedTransformer")
#ft_model = T5ForConditionalGeneration.from_pretrained("FineTunedTransformer")

# Load the file
file_path = 'jaccard_similarity_output.csv'
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
    lambda row: levenshtein_similarity_percentage(str(row['y_true_eng']), str(row['predicted_text'])), axis=1
)

# Save the updated dataframe to a new CSV file
output_file_path = 'jaccard_similarity_with_levenshtein_percentage.csv'
df.to_csv(output_file_path, index=False)

# Display the first few rows of the updated dataframe
print(df[['y_true_eng', 'predicted_text', 'levenshtein_similarity_percentage']].head())


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
file_path = 'jaccard_similarity_output.csv'
df = pd.read_csv(file_path)

# Apply the word-level Levenshtein similarity percentage calculation to each row
df['levenshtein_similarity_percentage_word_level'] = df.apply(
    lambda row: levenshtein_similarity_percentage_word_level(str(row['y_true_eng']), str(row['predicted_text'])), axis=1
)

# Save the updated dataframe to a new CSV file
output_file_path = 'jaccard_similarity_with_levenshtein_word_level.csv'
df.to_csv(output_file_path, index=False)

# Display the first few rows of the updated dataframe
print(df[['y_true_eng', 'predicted_text', 'levenshtein_similarity_percentage_word_level']].head())

# Calculate the aggregated similarity percentage (average of all rows)
aggregated_similarity_percentage = df['levenshtein_similarity_percentage_word_level'].mean()

# Print the aggregated similarity percentage
print(f"Aggregated Word-Level Levenshtein Similarity Percentage: {aggregated_similarity_percentage:.2f}%")
