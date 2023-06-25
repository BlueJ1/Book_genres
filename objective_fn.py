import numpy as np
import pandas as pd
import torch
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

# Set the number of folds
k_folds = 5

load_data = True
data_dir = "data"

if not os.path.isdir(data_dir):
    os.mkdir(data_dir)

if load_data:
    t = time()
    print("Loading data...")
    with open(os.path.join(data_dir, "vocab.json"), "r") as f:
        vocab = json.load(f)

    # tf_idf_data
    tf_idf_data = np.load(os.path.join(data_dir, "tf_idf_data.npy"))
    del_tf_idf_data_2 = np.load(os.path.join(data_dir, "del_tf_idf_data_2.npy"))
    del_tf_idf_data_1 = np.load(os.path.join(data_dir, "del_tf_idf_data_1.npy"))
    synonym_tf_idf_data_2 = np.load(os.path.join(data_dir, "synonym_tf_idf_data_2.npy"))
    synonym_tf_idf_data_1 = np.load(os.path.join(data_dir, "synonym_tf_idf_data_1.npy"))

    # labels
    categorical_genres = np.load(os.path.join(data_dir, "categorical_genres.npy"))
    del_categorical_genres_2 = np.load(os.path.join(data_dir, "del_categorical_genres_2.npy"))
    del_categorical_genres_1 = np.load(os.path.join(data_dir, "del_categorical_genres_1.npy"))
    synonym_categorical_genres_2 = np.load(os.path.join(data_dir, "synonym_categorical_genres_2.npy"))
    synonym_categorical_genres_1 = np.load(os.path.join(data_dir, "synonym_categorical_genres_1.npy"))
    print("Data loaded! Time elapsed: {:.2f}s.".format(time() - t))

    data_length = tf_idf_data.shape[0]
else:
    # Here, we load the dataset, removing duplicated instances and keeping only the first summary
    df = pd.read_csv("data.csv")[["summary", "genre"]]
    df = df.drop_duplicates(subset=["summary"], keep="first").reset_index(drop=True)

    test_df = df.sample(frac=test_size, random_state=42)
    train_df = df.drop(test_df.index).reset_index(drop=True)
    data_length = len(train_df)
    # test_one_hot_genres = to_categorical(test_df["genre"])

    del_p = 0.1
    synonym_p = 0.1
    aug_delete = nlpaw.RandomWordAug(action='delete', aug_p=del_p)
    aug_synonym = nlpaw.SynonymAug(aug_src='wordnet', aug_p=synonym_p)

    # We first augment the data
    t = time()
    print("Augmenting data...")
    del_augmented_df_2 = augment_text(train_df, aug_delete, num_threads=4, num_times=2)
    del_augmented_df_1 = augment_text(train_df, aug_delete, num_threads=4, num_times=1)

    synonym_augmented_df_2 = augment_text(train_df, aug_synonym, num_threads=4, num_times=2)
    synonym_augmented_df_1 = augment_text(train_df, aug_synonym, num_threads=4, num_times=1)

    print("Data augmented! - Time elapsed: {:.2f}s".format(time() - t))

    # We then proceed with the cleaning process
    t = time()
    print("Cleaning data...")
    corpus, vocab = clean_data(train_df["summary"])
    del_corpus_2, _ = clean_data(del_augmented_df_2["summary"])
    del_corpus_1, _ = clean_data(del_augmented_df_1["summary"])
    synonym_corpus_2, _ = clean_data(synonym_augmented_df_2["summary"])
    synonym_corpus_1, _ = clean_data(synonym_augmented_df_1["summary"])

    # This line gets all the frequencies of each word and sorts it in descending order
    frequencies = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
    # We choose to only consider the first 10000 most common non-stop words
    # print("Original vocabulary size: {}".format(len(frequencies)))
    vocab_size = 15000
    # Convert back to dictionary
    vocab = {x[0]: x[1] for x in frequencies[:vocab_size]}

    corpus = get_filtered_corpus(corpus, vocab.keys())
    del_corpus_2 = get_filtered_corpus(del_corpus_2, vocab.keys())
    del_corpus_1 = get_filtered_corpus(del_corpus_1, vocab.keys())
    synonym_corpus_2 = get_filtered_corpus(synonym_corpus_2, vocab.keys())
    synonym_corpus_1 = get_filtered_corpus(synonym_corpus_1, vocab.keys())
    print("Data cleaned! - Time elapsed: {:.2f}s".format(time() - t))

    t = time()
    print("Convert to tf-idf...")
    tf_idf_data = preprocess_tf_idf(corpus, vocab)
    del_tf_idf_data_2 = preprocess_tf_idf(del_corpus_2, vocab)
    del_tf_idf_data_1 = preprocess_tf_idf(del_corpus_1, vocab)
    synonym_tf_idf_data_2 = preprocess_tf_idf(synonym_corpus_2, vocab)
    synonym_tf_idf_data_1 = preprocess_tf_idf(synonym_corpus_1, vocab)

    categorical_genres = to_categorical(train_df["genre"])
    del_categorical_genres_2 = to_categorical(del_augmented_df_2["genre"])
    del_categorical_genres_1 = to_categorical(del_augmented_df_1["genre"])
    synonym_categorical_genres_2 = to_categorical(synonym_augmented_df_2["genre"])
    synonym_categorical_genres_1 = to_categorical(synonym_augmented_df_1["genre"])
    print("Data converted! - Time elapsed: {:.2f}s".format(time() - t))

    del corpus, del_corpus_2, del_corpus_1, synonym_corpus_2, synonym_corpus_1, df, train_df, \
        del_augmented_df_2, del_augmented_df_1, synonym_augmented_df_2, synonym_augmented_df_1

    t = time()
    print("Saving data...")
    # We save the data to disk, as it takes a long time to augment the data
    with open(os.path.join(data_dir, "vocab.json"), "w+") as f:
        json.dump(vocab, f)

    np.save(os.path.join(data_dir, "tf_idf_data.npy"), tf_idf_data)
    np.save(os.path.join(data_dir, "del_tf_idf_data_2.npy"), del_tf_idf_data_2)
    np.save(os.path.join(data_dir, "del_tf_idf_data_1.npy"), del_tf_idf_data_1)
    np.save(os.path.join(data_dir, "synonym_tf_idf_data_2.npy"), synonym_tf_idf_data_2)
    np.save(os.path.join(data_dir, "synonym_tf_idf_data_1.npy"), synonym_tf_idf_data_1)

    np.save(os.path.join(data_dir, "categorical_genres.npy"), categorical_genres)
    np.save(os.path.join(data_dir, "del_categorical_genres_2.npy"), del_categorical_genres_2)
    np.save(os.path.join(data_dir, "del_categorical_genres_1.npy"), del_categorical_genres_1)
    np.save(os.path.join(data_dir, "synonym_categorical_genres_2.npy"), synonym_categorical_genres_2)
    np.save(os.path.join(data_dir, "synonym_categorical_genres_1.npy"), synonym_categorical_genres_1)
    print("Data saved! - Time elapsed: {:.2f}s".format(time() - t))

# Define the K-fold Cross Validator
kfold = KFold(n_splits=k_folds, shuffle=True)

iteration = 0


def objective(hyperparameters):
    global iteration

    # retrieving hyperparameters
    model_size = list(hyperparameters["model_size"])
    print(hyperparameters)
    learning_rate = hyperparameters["learning_rate"]
    l2 = hyperparameters["l2"]
    dropout = hyperparameters["dropout"]

    # data augmentation
    del_num_times = int(hyperparameters["del_num_times"])
    synonym_num_times = int(hyperparameters["synonym_num_times"])

    train_losses = []
    validation_acc = []
    validation_f1 = []
    validation_losses = []
    confusion_matrices = []

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, validation_ids) in enumerate(kfold.split(np.arange(data_length))):
        validation_tf_idf_data = tf_idf_data[validation_ids]
        validation_one_hot_genres = categorical_genres[validation_ids]

        train_tf_idf_data = tf_idf_data[train_ids]
        if del_num_times == 1:
            train_tf_idf_data = np.concatenate([train_tf_idf_data, del_tf_idf_data_1[train_ids]], axis=0)
        elif del_num_times == 2:
            train_tf_idf_data = np.concatenate([train_tf_idf_data, del_tf_idf_data_2[train_ids]], axis=0)
        if synonym_num_times == 1:
            train_tf_idf_data = np.concatenate([train_tf_idf_data, synonym_tf_idf_data_1[train_ids]], axis=0)
        elif synonym_num_times == 2:
            train_tf_idf_data = np.concatenate([train_tf_idf_data, synonym_tf_idf_data_2[train_ids]], axis=0)

        train_one_hot_genres = categorical_genres[train_ids]
        if del_num_times == 1:
            train_one_hot_genres = np.concatenate([train_one_hot_genres, del_categorical_genres_1[train_ids]], axis=0)
        elif del_num_times == 2:
            train_one_hot_genres = np.concatenate([train_one_hot_genres, del_categorical_genres_2[train_ids]], axis=0)
        if synonym_num_times == 1:
            train_one_hot_genres = np.concatenate([train_one_hot_genres, synonym_categorical_genres_1[train_ids]],
                                                  axis=0)
        elif synonym_num_times == 2:
            train_one_hot_genres = np.concatenate([train_one_hot_genres, synonym_categorical_genres_2[train_ids]],
                                                  axis=0)

        # print(train_tf_idf_data.shape)
        # print(train_one_hot_genres.shape)

        print(f'FOLD {fold}')
        print('--------------------------------')

        train_dataset = TF_IDF_Dataset(torch.Tensor(train_tf_idf_data).to(device),
                                       torch.LongTensor(train_one_hot_genres).to(device))
        validation_dataset = TF_IDF_Dataset(torch.Tensor(validation_tf_idf_data).to(device),
                                            torch.LongTensor(validation_one_hot_genres).to(device))

        train_loader = torch.utils.data.DataLoader(train_dataset, **loader_args)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, **loader_args)

        train_loss = []

        # Init the neural network
        network = MLP([len(vocab)] + model_size + [10], dropout)
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
                # inputs = inputs.to(device)
                # targets = torch.argmax(targets.data, 1)

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
            # print(f'Loss after epoch {epoch+1}: {epoch_loss}')

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
        validation_loss = 0

        with torch.no_grad():
            network.eval()
            # Iterate over the test data and generate predictions
            for data in validation_loader:
                # Get inputs
                inputs, targets = data
                # inputs = inputs.to(device)
                # targets = torch.argmax(targets.to(device).data, 1)

                # Generate outputs
                outputs = network(inputs)

                # Compute loss
                loss = criterion(outputs, targets)
                validation_loss += loss.item()

                # Set total and correct
                predicted = torch.argmax(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                y_true.extend(targets.tolist())
                y_pred.extend(predicted.tolist())

        validation_loss /= len(validation_loader)
        validation_losses.append(validation_loss)

        # Print accuracy and f1 score metrics
        accuracy = 100.0 * correct / total
        f1 = f1_score(y_true, y_pred, average='weighted')
        validation_acc.append(accuracy)
        validation_f1.append(f1)

        print('Accuracy for fold %d: %.1f %%' % (fold, accuracy))
        print('F1 score for fold %d: %.1f %%' % (fold, 100 * f1))
        print('--------------------------------')

        confusion = confusion_matrix(y_true, y_pred)

        train_losses.append(train_loss)
        confusion_matrices.append(confusion)

    """hyperparameter_df.loc[hyperparameter_set.Index, "train_losses"] = train_losses
    hyperparameter_df.loc[hyperparameter_set.Index, "validation_accs"] = validation_acc
    hyperparameter_df.loc[hyperparameter_set.Index, "validation_f1s"] = validation_f1
    hyperparameter_df.loc[hyperparameter_set.Index, "confusion_matrices"] = confusion_matrices

    # Saving the results
    hyperparameter_df.to_csv("hyperparameter_results.csv")"""

    return {
        'loss': -np.mean(validation_f1),  # remember: HpBandSter always minimizes!
        'status': STATUS_OK,
        'hyperparameters': hyperparameters,
        'train_losses': train_losses,
        'validation_accs': validation_acc,
        'validation_f1s': validation_f1,
        'confusion_matrices': confusion_matrices,
        'iteration': iteration
    }
