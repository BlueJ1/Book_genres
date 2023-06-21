import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, confusion_matrix

from process_data import clean_data, get_filtered_corpus, to_categorical, TF_IDF_Dataset
from tf_idf import preprocess_tf_idf
from mlp import MLP
import nlpaug.augmenter.word as nlpaw
from data_augmentation import augment_text

# Here, we load the dataset, removing duplicated instances and keeping only the first summary
df = pd.read_csv("data.csv")[["summary", "genre"]]
df = df.drop_duplicates(subset=["summary"], keep="first").reset_index(drop=True)

test_size = 0.2
test_df = df.sample(frac=test_size, random_state=42)
train_df = df.drop(test_df.index).reset_index(drop=True)
test_one_hot_genres = to_categorical(test_df["genre"])

# Set fixed random number seed
torch.manual_seed(42)

# Here we set how many processes we will run at the same time
num_workers = 2 if torch.cuda.is_available() else 0

n_epochs = 1000
batch_size = 256
patience = 20

if torch.cuda.is_available() or torch.backends.mps.is_available():
    loader_args = dict(shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
else:
    loader_args = dict(shuffle=True, batch_size=128)

# Set the number of folds
k_folds = 5
# Define the K-fold Cross Validator
kfold = KFold(n_splits=k_folds, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print("Using device {}.".format(device))

# hyperparameters = [{"model": [len(vocab), 512, 256, 32, 10], "lr": 2e-3, "l2": 1e-4, "dropout": 0.25}]
hyperparameter_df = pd.DataFrame([], columns=["model_size", "lr", "l2", "dropout", "num_times", "del_p", "synonym_p",
                                              "train_losses", "validation_acc", "validation_f1", "confusion_matrices"])
hyperparameter_df.loc[-1] = [[512, 32], 2e-3, 1e-4, 0.25, 1, 0.1, 0.1, [], [], [], []]
hyperparameter_df.loc[-1] = [[1024, 256, 64], 1e-3, 3e-5, 0.4, 2, 0.1, 0.1, [], [], [], []]


for hyperparameter_set in hyperparameter_df.itertuples():
    # retrieving hyperparameters
    model_size = hyperparameter_set.model_size
    learning_rate = hyperparameter_set.lr
    l2 = hyperparameter_set.l2
    dropout = hyperparameter_set.dropout

    Train_losses = hyperparameter_set.train_losses
    validation_acc = hyperparameter_set.validation_acc
    validation_f1 = hyperparameter_set.validation_f1
    confusion_matrices = hyperparameter_set.confusion_matrices

    # apply data augmentation
    num_times = hyperparameter_set.num_times
    del_p = hyperparameter_set.del_p
    synonym_p = hyperparameter_set.synonym_p
    aug_delete = nlpaw.RandomWordAug(action='delete', aug_p=del_p)
    aug_synonym = nlpaw.SynonymAug(aug_src='wordnet', aug_p=synonym_p)

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, validation_ids) in enumerate(kfold.split(np.arange(len(train_df)))):
        validation_df = train_df.iloc[validation_ids]
        train_df_fold = train_df.iloc[train_ids]
        print(f'FOLD {fold}')
        print('--------------------------------')

        if del_p > 0:
            del_augmented_df = augment_text(train_df_fold, aug_delete, num_threads=4, num_times=num_times)
        else:
            del_augmented_df = pd.DataFrame([], columns=["summary", "genre"])
        if synonym_p > 0:
            synonym_augmented_df = augment_text(train_df_fold, aug_synonym, num_threads=4, num_times=num_times)
        else:
            synonym_augmented_df = pd.DataFrame([], columns=["summary", "genre"])

        augmented_df = pd.concat([train_df_fold, del_augmented_df, synonym_augmented_df], ignore_index=True)

        # We then proceed with the cleaning process
        train_corpus, vocab = clean_data(augmented_df["summary"].copy())
        validation_corpus, _ = clean_data(validation_df["summary"].copy())

        # This line gets all the frequencies of each word and sorts it in descending order
        frequencies = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
        # We choose to only consider the first 10000 most common non-stop words
        VOCAB_SIZE = 10000
        # Convert back to dictionary
        vocab = {x[0]: x[1] for x in frequencies[:VOCAB_SIZE]}

        train_corpus = get_filtered_corpus(train_corpus, vocab.keys())
        validation_corpus = get_filtered_corpus(validation_corpus, vocab.keys())

        # Here we store the TF-IDF values of each corpus from the vocabulary
        train_tf_idf_data = preprocess_tf_idf(train_corpus, vocab)
        validation_tf_idf_data = preprocess_tf_idf(validation_corpus, vocab)
        # test_tf_idf_data = preprocess_tf_idf(test_corpus, vocab)

        train_one_hot_genres = to_categorical(augmented_df["genre"])
        validation_one_hot_genres = to_categorical(validation_df["genre"])

        train_dataset = TF_IDF_Dataset(torch.from_numpy(train_tf_idf_data), torch.from_numpy(train_one_hot_genres))
        validation_dataset = TF_IDF_Dataset(torch.from_numpy(validation_tf_idf_data),
                                            torch.from_numpy(validation_one_hot_genres))

        train_loader = torch.utils.data.DataLoader(train_dataset, **loader_args)
        validation_loader = torch.utils.data.DataLoader(validation_dataset, **loader_args)

        Train_loss = []

        # Init the neural network
        # model_size = [len(vocab)] + model_size + [10]
        network = MLP([len(vocab)] + model_size + [10], dropout)
        network.to(device)

        # Initialize optimizer
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate, weight_decay=l2)
        criterion = torch.nn.CrossEntropyLoss()

        # Initialize the early stopping variables
        best_epoch_loss = np.inf
        epochs_without_improvement = 0
        confusion = np.zeros((10, 10), dtype=int)

        # Run the training loop for defined number of epochs
        for epoch in range(0, n_epochs):

            # Print epoch
            print(f'Starting epoch {epoch+1}')

            # Set current loss value
            current_loss = 0.0

            network.train()
            # Iterate over the DataLoader for training data
            for i, data in enumerate(train_loader, 0):
                # print(f'Batch {i+1}')

                # Get inputs
                inputs, targets = data
                inputs = inputs.to(device)
                targets = torch.argmax(targets.to(device).data, 1)

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
            Train_loss.append(epoch_loss)
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

        with torch.no_grad():
            network.eval()
            # Iterate over the test data and generate predictions
            for data in validation_loader:

                # Get inputs
                inputs, targets = data
                inputs = inputs.to(device)
                targets = torch.argmax(targets.to(device).data, 1)

                # Generate outputs
                outputs = network(inputs)

                # Set total and correct
                predicted = torch.argmax(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                y_true.extend(targets.tolist())
                y_pred.extend(predicted.tolist())

        # Print accuracy
        accuracy = 100.0 * correct / total
        f1 = f1_score(y_true, y_pred, average='weighted')
        validation_acc.append(accuracy)
        validation_f1.append(f1)

        print('Accuracy for fold %d: %.1f %%' % (fold, accuracy))
        print('F1 score for fold %d: %.1f %%' % (fold, 100 * f1))
        print('--------------------------------')

        confusion = confusion_matrix(y_true, y_pred)

        Train_losses.append(Train_loss)
        confusion_matrices.append(confusion)

    hyperparameter_df.loc[hyperparameter_set.Index, "train_losses"] = Train_losses
    hyperparameter_df.loc[hyperparameter_set.Index, "validation_accs"] = validation_acc
    hyperparameter_df.loc[hyperparameter_set.Index, "validation_f1s"] = validation_f1
    hyperparameter_df.loc[hyperparameter_set.Index, "confusion_matrices"] = confusion_matrices

    # Saving the results
    hyperparameter_df.to_csv("hyperparameter_results.csv")
