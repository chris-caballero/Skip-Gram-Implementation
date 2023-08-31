import sys
sys.path.append('../utils')

import numpy as np
from utils import *
from visutils import * 
from model_utils import *

WINDOW_SIZE = 2
EMBEDDING_DIM = 32
EPOCHS = 10
LR = 0.005
TARGET_ID = 0

def main():
    DATA_DIR = '../../data/'
    FILE = 'corpus.txt'

    filename = DATA_DIR + FILE

    try:
        corpus = get_data(filename)
        words = process_data(corpus)

        tokenizer = Tokenizer()
        tokenizer.fit(words)
        encoding = encode_words(words, tokenizer)

        dataset = create_dataset(encoding, tokenizer, window_size=WINDOW_SIZE)
    except:
        print('Error creating the dataset')
        exit(1)

    try:
        model = SkipGramModel(
            vocab_size=len(tokenizer.vocab), 
            embedding_dim=EMBEDDING_DIM,
            learning_rate=LR,
            target_id=TARGET_ID
        )
    except:
        print('Error creating the model')
        exit(1)

    try:
        train(model, epochs=EPOCHS, training_data=dataset, verbose=False)
    except:
        print('Error in training the model')
        exit(1)

    try:
        ani = visualize_vectors_over_time(model.vectors_over_time, tokenizer.index_to_key, num_words=30)
        ground_truth = get_context_distribution(dataset, TARGET_ID, len(tokenizer.vocab))
        ani = visualize_predictions_over_time(model.predictions_over_time, ground_truth, tokenizer.index_to_key)
    except:
        print('Error creating visualizations')
        exit(1)

    print('Completed Successfully!')

  
if __name__ == "__main__":
    main()