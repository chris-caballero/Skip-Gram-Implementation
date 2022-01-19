import numpy as np
from skip_gram import *
from skip_gram_util import * 

def main():
    file = 'word_dataset.txt'
    data, contents = get_data(file)
    word_to_int, int_to_word, vocabulary = encode_words(data)
    # print('Unique: ', len(vocabulary))
    # print('Total: ', len(data), '\n')

    window_size = 2
    num_features = 10
    training_data = generate_training_data(data, window_size, word_to_int)
    x_train = np.array([pair[0] for pair in training_data])
    y_train = np.array([pair[1] for pair in training_data])

    embedding_matrix = np.random.uniform(-1, 1, (len(vocabulary), num_features))
    weights_out = np.random.uniform(-1, 1, (num_features, len(vocabulary)))

    epochs = 500
    learning_rate = 0.01
    test_word = word_to_int['every']
    vectors_over_time, pred_over_time = train(x_train, y_train, embedding_matrix, weights_out, epochs, learning_rate, test_word)

    ground_truth_table = ground_truth(len(vocabulary), training_data, window_size)


    x = [i for i in range(len(vocabulary))]
    y_true = ground_truth_table[test_word]
    y_true = sorted(y_true.items(), key=lambda item: item[1], reverse=True)

    for i in range(len(y_true)):
        y_true[i] = y_true[i][1]

    sort_predictions(pred_over_time)

    mode = 0
    print_info(list(word_to_int.keys()), embedding_matrix, weights_out, word_to_int, int_to_word)
    visualize_data(vectors_over_time, x, y_true, pred_over_time, epochs, int_to_word, mode)
  
if __name__ == "__main__":
    main()