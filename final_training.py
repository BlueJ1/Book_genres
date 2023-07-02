import numpy as np
import pandas as pd
import torch
import math
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, confusion_matrix
from hyperopt import STATUS_OK

from process_data import clean_data, get_filtered_corpus, to_categorical, TF_IDF_Dataset
from tf_idf import preprocess_tf_idf
from mlp import MLP
import nlpaug.augmenter.word as nlpaw
from data_augmentation import augment_text
import os
import json

from time import time
from tqdm import tqdm

# Set fixed random number seed
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print("Using device {}.".format(device))

n_epochs = 400
batch_size = 512
patience = 20
test_size = 0.2

if torch.cuda.is_available() or torch.backends.mps.is_available():
    loader_args = dict(shuffle=True, batch_size=batch_size)
else:
    loader_args = dict(shuffle=True, batch_size=64)

load_data = True
data_dir = "data"

if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

if load_data:
    t = time()
    print("Loading data...")
    with open(os.path.join(data_dir, "vocab.json"), "r") as f:
        train_vocab = json.load(f)

    # tf_idf_data
    train_tf_idf_data = np.load(os.path.join(data_dir, "train_tf_idf_data.npy"))
    synonym_tf_idf_data_1 = np.load(os.path.join(data_dir, "synonym_tf_idf_data_1.npy"))
    test_tf_idf_data = np.load(os.path.join(data_dir, "test_tf_idf_data.npy"))

    # labels
    train_categorical_genres = np.load(os.path.join(data_dir, "train_categorical_genres.npy"))
    synonym_categorical_genres_1 = np.load(os.path.join(data_dir, "synonym_categorical_genres_1.npy"))
    test_categorical_genres = np.load(os.path.join(data_dir, "test_categorical_genres.npy"))
    print("Data loaded! Time elapsed: {:.2f}s.".format(time() - t))

    data_length = train_tf_idf_data.shape[0]
else:
    # Here, we load the dataset, removing duplicated instances and keeping only the first summary
    df = pd.read_csv("data.csv")[["summary", "genre"]]
    df = df.drop_duplicates(subset=["summary"], keep="first").reset_index(drop=True)

    test_df = df.sample(frac=test_size, random_state=42)
    train_df = df.drop(test_df.index).reset_index(drop=True)
    data_length = len(train_df)

    del_p = 0.1
    synonym_p = 0.1
    aug_delete = nlpaw.RandomWordAug(action='delete', aug_p=del_p)
    aug_synonym = nlpaw.SynonymAug(aug_src='wordnet', aug_p=synonym_p)

    # We first augment the data
    t = time()
    print("Augmenting data...")
    synonym_augmented_df_1 = augment_text(train_df, aug_synonym, num_threads=4, num_times=1)
    print("Data augmented! - Time elapsed: {:.2f}s".format(time() - t))

    # We then proceed with the cleaning process
    t = time()
    print("Cleaning data...")
    train_corpus, train_vocab = clean_data(train_df["summary"])
    test_corpus, test_vocab = clean_data(test_df["summary"])
    synonym_corpus_1, _ = clean_data(synonym_augmented_df_1["summary"])

    # This line gets all the frequencies of each word and sorts it in descending order
    train_frequencies = sorted(train_vocab.items(), key=lambda x: x[1], reverse=True)
    test_frequencies = sorted(test_vocab.items(), key=lambda x: x[1], reverse=True)

    # We choose to only consider the first 15000 most common non-stop words
    vocab_size = 15000
    # Convert back to dictionary
    train_vocab = {x[0]: x[1] for x in train_frequencies[:vocab_size]}
    test_vocab = {x[0]: x[1] for x in test_frequencies[:vocab_size]}

    train_corpus = get_filtered_corpus(train_corpus, train_vocab.keys())
    test_corpus = get_filtered_corpus(test_corpus, test_vocab.keys())
    synonym_corpus_1 = get_filtered_corpus(synonym_corpus_1, train_vocab.keys())
    print("Data cleaned! - Time elapsed: {:.2f}s".format(time() - t))

    t = time()
    print("Convert to tf-idf...")
    train_tf_idf_data = preprocess_tf_idf(train_corpus, train_vocab)
    test_tf_idf_data = preprocess_tf_idf(test_corpus, test_vocab)
    synonym_tf_idf_data_1 = preprocess_tf_idf(synonym_corpus_1, train_vocab)

    # Get the unique genre labels in a sorted order
    unique_genres = df["genre"].unique()
    unique_genres.sort()

    # Create a mapping between genre labels and category indices
    genre_to_category = {genre: i for i, genre in enumerate(unique_genres)}

    # Perform one-hot encoding using the genre-to-category mapping
    train_categorical_genres = to_categorical(train_df["genre"].map(genre_to_category))
    test_categorical_genres = to_categorical(test_df["genre"].map(genre_to_category))
    synonym_categorical_genres_1 = to_categorical(synonym_augmented_df_1["genre"].map(genre_to_category))

    print("Data converted! - Time elapsed: {:.2f}s".format(time() - t))

    del train_corpus, synonym_corpus_1, df, train_df, \
        synonym_augmented_df_1

    t = time()
    print("Saving data...")
    # We save the data to disk, as it takes a long time to augment the data
    with open(os.path.join(data_dir, "vocab.json"), "w+") as f:
        json.dump(train_vocab, f)

    np.save(os.path.join(data_dir, "train_tf_idf_data.npy"), train_tf_idf_data)
    np.save(os.path.join(data_dir, "test_tf_idf_data.npy"), test_tf_idf_data)
    np.save(os.path.join(data_dir, "synonym_tf_idf_data_1.npy"), synonym_tf_idf_data_1)

    np.save(os.path.join(data_dir, "train_categorical_genres.npy"), train_categorical_genres)
    np.save(os.path.join(data_dir, "test_categorical_genres.npy"), test_categorical_genres)
    np.save(os.path.join(data_dir, "synonym_categorical_genres_1.npy"), synonym_categorical_genres_1)
    print("Data saved! - Time elapsed: {:.2f}s".format(time() - t))

iteration = 0

# retrieving hyperparameter
#RAND
model_size = [32]
learning_rate = 0.0003557400109264205
l2 = 6.336724645751509e-6
dropout = 0.1101374997984408

#TPE
# model_size = [128]
# learning_rate = 0.00032449781569679487
# l2 = 5.772855766644525e-6
# dropout = 0.3154875519178951

# data augmentation
synonym_num_times = 1

train_losses = []
test_acc = []
test_f1 = []
test_losses = []
confusion_matrices = []

train_tf_idf_data = np.concatenate([train_tf_idf_data, synonym_tf_idf_data_1], axis=0)
train_one_hot_genres = train_categorical_genres
train_one_hot_genres = np.concatenate([train_one_hot_genres, synonym_categorical_genres_1], axis=0)

train_dataset = TF_IDF_Dataset(torch.Tensor(train_tf_idf_data).to(device),
                               torch.LongTensor(train_one_hot_genres).to(device))
test_dataset = TF_IDF_Dataset(torch.Tensor(test_tf_idf_data).to(device),
                                    torch.LongTensor(test_categorical_genres).to(device))

train_loader = torch.utils.data.DataLoader(train_dataset, **loader_args)
test_loader = torch.utils.data.DataLoader(test_dataset, **loader_args)

train_loss = []

# Init the neural network
network = MLP([len(train_vocab)] + model_size + [10], dropout)
network.to(device)

# Initialize optimizer
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=l2)
criterion = torch.nn.CrossEntropyLoss()

# Initialize the early stopping variables
best_epoch_loss = np.inf
epochs_without_improvement = 0
# confusion = np.zeros((10, 10), dtype=int)

# Run the training loop for defined number of epochs
for epoch in range(0, n_epochs):
    # Print epoch
    # print(f'Starting epoch {epoch+1}')

    # Set current loss value
    current_loss = 0.0

    network.train()
    # print("Training...")
    # Iterate over the DataLoader for training data
    for i, data in enumerate(train_loader, 0):
        # print(f'Batch {i+1}')

        # Get inputs
        inputs, targets = data

        # Zero the gradients
        optimizer.zero_grad()

        # Perform forward pass
        outputs = network(inputs)

        # Compute loss
        loss = criterion(outputs, targets)

        # Perform backward pass
        loss.backward()

        # Perform optimization
        optimizer.step()

        current_loss += loss.item()

    epoch_loss = current_loss / len(train_loader)
    train_loss.append(epoch_loss)
    print(f'Loss after epoch {epoch+1}: {epoch_loss}')

    # Check if the F1 score has improved
    if epoch_loss < best_epoch_loss:
        best_epoch_loss = epoch_loss
        epochs_without_improvement = 0
    else:
        epochs_without_improvement += 1

    # Check for early stopping
    if epochs_without_improvement >= patience:
        print("Early stopping after %d epochs" % (epoch + 1))
        break

# Evaluation for this fold
correct, total = 0, 0
y_true, y_pred = [], []
test_loss = 0

with torch.no_grad():
    network.eval()
    # Iterate over the test data and generate predictions
    for data in test_loader:
        # Get inputs
        inputs, targets = data
        # inputs = inputs.to(device)
        # targets = torch.argmax(targets.to(device).data, 1)

        # Generate outputs
        outputs = network(inputs)

        # Compute loss
        loss = criterion(outputs, targets)
        test_loss += loss.item()

        # Set total and correct
        predicted = torch.argmax(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
        y_true.extend(targets.tolist())
        y_pred.extend(predicted.tolist())

test_loss /= len(test_loader)
test_losses.append(test_loss)

# Print accuracy and f1 score metrics
accuracy = 100.0 * correct / total
print("Correct : %d" % correct)
print("Total : %d" % total )
f1 = f1_score(y_true, y_pred, average='weighted')
test_acc.append(accuracy)
test_f1.append(f1)

print('Accuracy : %.1f %%' % (accuracy))
print('F1 score : %.1f %%' % ( 100 * f1))
print('--------------------------------')

confusion = confusion_matrix(y_true, y_pred)

train_losses.append(train_loss)
confusion_matrices.append(confusion)

data = {
    'loss': -np.mean(test_f1),
    'status': STATUS_OK,
    'hyperparameters': "rand_best",
    'train_losses': train_losses,
    'test_accs': test_acc,
    'test_f1s': test_f1,
    'confusion_matrices': confusion_matrices,
    'iteration': iteration
}

df = pd.DataFrame(data)
csv_file = data_dir + '/rand_best_final.csv'
df.to_csv(csv_file, index=False)
