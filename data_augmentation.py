import pandas as pd
from tqdm import tqdm


def augment_sentence(sentence, aug, num_threads):
    """""""""
    Constructs a new sentence via text augmentation.

    Input:
        - sentence:     A string of text
        - aug:          An augmentation object defined by the nlpaug library
        - num_threads:  Integer controlling the number of threads to use if
                        augmenting text via CPU
    Output:
        - A string of text that been augmented
    """""""""
    x = aug.augment(sentence, num_thread=num_threads)
    x = ' '.join(x)
    return x


def augment_text(df, aug, num_threads, num_times, genres_to_augment: list = None):
    """""""""
    Takes a pandas DataFrame and augments its text data.

    Input:
        - df:            A pandas DataFrame containing the columns:
                                - 'summary' containing strings of text to augment.
                                - 'genre' target variable containing genres.
        - aug:           Augmentation object defined by the nlpaug library.
        - num_threads:   Integer controlling number of threads to use if augmenting
                         text via CPU
        - num_times:     Integer representing the number of times to augment text.
        - genres_to_augment: names of the genres that should be augmented
        - num_new: number of new sentences to be kept
    Output:
        - df:            Copy of the same pandas DataFrame with augmented data 
                         appended to it and with rows randomly shuffled
    """""""""

    if genres_to_augment is None:
        genres_to_augment = df.genre.unique()

    # Get rows of data to augment
    for i in range(len(genres_to_augment)):
        genre = genres_to_augment[i]
        to_augment = df[df['genre'] == genre]
        to_augment_x = to_augment['summary']
        to_augment_y = genre

        # Build up dictionary containing augmented data
        aug_dict = {'summary': [], 'genre': to_augment_y}
        for _ in range(num_times):
            aug_x = [augment_sentence(x, aug, num_threads) for x in to_augment_x]
            aug_dict['summary'].extend(aug_x)

        # Build DataFrame containing augmented data
        aug_df = pd.DataFrame.from_dict(aug_dict)
        df = pd.concat([df, aug_df], ignore_index=True)

    return df
