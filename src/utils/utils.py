import re
import numpy as np
from math import exp
from sklearn.decomposition import TruncatedSVD
from collections import defaultdict

# process contents of a text file into list of words
def get_data(filename):
    contents = []
    with open(filename, 'r') as f:
        lines = f.read()
    
    return lines

# def process_data(filename):

# integer encoding so each word gets a unique index
def encode_words(data):
    # we want unique words
    vocabulary = list(set(data))
    word_to_int = dict((word, i) for i, word in enumerate(vocabulary))
    int_to_word = dict((i, word) for i, word in enumerate(vocabulary))
    return word_to_int, int_to_word, vocabulary

# prints the training data in string representation
def print_training_data(training_data, int_to_word):
    for pair in training_data:
        print('Target: ', int_to_word[pair[0][1]])
        print('Context: ', pair[1][1])
        print('\n')

# prints the training data in vector representation
def print_training_data_raw(training_data):
    for pair in training_data:
        print('Target: ', pair[0], '\nContext: ', pair[1])
        print('\n')


def print_info(testset, embedding_matrix, weights_out, word_to_int, int_to_word):
    from utils.utils import forward_pass, cos_similarity_dict
    num_similar = 10
    for word in testset:
        h, y, s = forward_pass(
            word_to_int[word], embedding_matrix, weights_out)
        similarities_in, similarities_out = cos_similarity_dict(
            word, embedding_matrix, weights_out.T, word_to_int, int_to_word)
        # print target word, context prediction with certainty, and a few of the most similar word vectors
        print('{:<15}'.format('Target:'), word)
        print('{:<15}'.format('Prediction:'), int_to_word[np.argmax(y)])
        print('{:<15}'.format('Probability:'), y[np.argmax(y)].round(4))
        print('{:<15}'.format('Most similar Input:'),
              list(similarities_in)[:num_similar])
        print('{:<15}'.format('Most similar Output:'),
              list(similarities_out)[:num_similar], '\n')


# makes the one-hot vector representation for a given word
def generate_one_hot(word, word_to_int):
    one_hot = np.zeros(len(word_to_int))
    one_hot[word_to_int[word]] = 1
    return one_hot, word_to_int[word]

# gets the one-hot vector for the target and context words
def get_target_context_pairs(target, context, word_to_int):
    target_vector = generate_one_hot(target, word_to_int)
    context_vectors = list()
    for w in context:
        context_vectors.append(generate_one_hot(w, word_to_int))
    return target_vector, context_vectors

# generates the training data (x_train, y_train)
def generate_training_data(data, window_size, word_to_int):
    training_data = []
    for (target_idx, target)  in enumerate(data):
        left = -window_size
        right = window_size
        if window_size > target_idx:
            left = -target_idx
        if window_size + target_idx > len(data) - 1:
            right = len(data) - target_idx - 1

        context = [data[target_idx + i] for i in range(left, right+1) if i != 0]
        target_vector, context_vectors = get_target_context_pairs(target, context, word_to_int)
        training_data.append([target_vector, context_vectors])

    return training_data

# applies softmax to the output layer
def softmax(score):
    sum_exp = sum(np.exp(score))
    for i in range(len(score)):
        score[i] = exp(score[i]) / sum_exp
    return score

'''
def softmax(score):
    exp_scores = np.exp(score - np.max(score))  # Improved numerical stability
    return exp_scores / exp_scores.sum()


'''

# evaluates the loss function for the predicted score
def loss(context, score):
    loss = len(context)*np.log(np.sum(np.exp(score)))
    for context_word in context:
        loss -= score[context_word[1]]
    return loss

# generate prediction of context on current target word
def forward_pass(target_idx, embedding_matrix, weight_matrix_out):
    # choosing the right word vector from our embedding matrix
    hidden_layer = embedding_matrix[target_idx]
    # pass from hidden layer to output layer creates scores for each word in vocab
    score = np.dot(weight_matrix_out.T, hidden_layer)
    # prediction is a distribution of how likely a given word is in the context
    # this is typical softmax classification
    y_pred = softmax(score.copy())

    return hidden_layer, y_pred, score

# gradient of loss function with respect to scores is the prediction error
def prediction_error(context, y_pred):
    # error is going to be y_pred - 1 at each context word index
    # otherwise it is y_pred
    error = len(context)*y_pred
    for context_word in context:
        error[context_word[1]] -= 1
    return error

# update the hidden->output weights with gradient descent 
def update_weights_out(hidden_layer, weights_out, pred_errors, learning_rate):
    weights_out_T = weights_out.T
    for j in range(len(weights_out_T)):
        weights_out_T[j] -= learning_rate*pred_errors[j]*hidden_layer
    # returns the updates weights, must set the old weights to this
    weights_out[:] = weights_out_T.T


'''
def update_weights_out(hidden_layer, weights_out, pred_errors, learning_rate):
    weights_out -= learning_rate * np.outer(hidden_layer, pred_errors)

def update_weights_in(target, weights_in, weights_out, pred_errors, learning_rate):
    EH = np.dot(weights_out, pred_errors)
    weights_in[target[1]] -= learning_rate * EH
'''

# update the input->hidden weights with gradient descent
def update_weights_in(target, weights_in, weights_out, pred_errors, learning_rate):
    EH = np.dot(weights_out, pred_errors)
    weights_in[target[1]] -= learning_rate*EH.T

# update the weights with gradient descent
def backward_propagation(target, context, hidden_layer, y_pred, weights_in, weights_out, learning_rate):
    error = prediction_error(context, y_pred)
    update_weights_out(hidden_layer, weights_out, error, learning_rate)
    update_weights_in(target, weights_in, weights_out, error, learning_rate)

# trains the network over epochs
def train(x_train, y_train, weights_in, weights_out, epochs, learning_rate, focus = 0, verbose=1):
    vectors_over_time = list()
    pred_over_time = list()
    for i in range(epochs):
        for (target, context) in zip(x_train, y_train):
            hidden_layer, y_pred, score = forward_pass(target[1], weights_in, weights_out)
            backward_propagation(target, context, hidden_layer, y_pred, weights_in, weights_out, learning_rate)
        if i % 10 == 0:
            x = weights_in.T.copy()
            svd = TruncatedSVD()
            svd.fit(x)
            vectors_over_time.append(x)
            pred_over_time.append(forward_pass(focus, weights_in, weights_out)[1])
        if verbose == 1 and i % 50 == 0:
            print(loss(context, score))
            
    return vectors_over_time, pred_over_time
    
# cosine similarity is measure of how small the angel between two vectors (x, y) is
def cos_dist(x, y):
    return np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))

# creates cosine similarity dictionary sorted by most similar
# given a word, makes a dictionary with each other word in vocab and similarity b/w them
def cos_similarity_dict(word, embedding_matrix, weights_out, word_to_int, int_to_word, mode=0):
    word_vector = embedding_matrix[word_to_int[word]]

    if mode == 1:
        similarities = {
            int_to_word[i] : cos_dist(word_vector, embedding_matrix[i]) for i in range(len(embedding_matrix))
        }
    elif mode == 2:
        similarities = {
            int_to_word[i] : cos_dist(word_vector, weights_out[i]) for i in range(len(weights_out))
        }
    else:
        similarities_in = {int_to_word[i]: cos_dist(word_vector, embedding_matrix[i]) for i in range(len(embedding_matrix))}
        similarities_out = {int_to_word[i]: cos_dist(word_vector, weights_out[i]) for i in range(len(weights_out))}

        return  sorted(similarities_in.items(), key=lambda item: item[1], reverse=True), \
                sorted(similarities_out.items(), key=lambda item: item[1], reverse=True)

    return sorted(similarities.items(), key=lambda item: item[1], reverse=True)

def target_to_total_context(training_data):
    target_total_context = dict()
    num_appearances = dict()
    for (target, context) in training_data:
        if target[1] in target_total_context:
            num_appearances[target[1]] += 1
        else:
            num_appearances[target[1]] = 1
        for context_word in context:
            if target[1] in target_total_context:
                target_total_context[target[1]] += context_word[0]
            else:
                target_total_context[target[1]] = context_word[0]
    return target_total_context, num_appearances

def ground_truth(vocab_size, training_data, window_size):
    target_total_context, num_appearances = target_to_total_context(training_data)
    ground_truth = dict()
    for i in range(vocab_size):
        y_true = dict()
        total_context = target_total_context[i]
        total = 2 * num_appearances[i] * window_size
        for j in range(len(total_context)):
            y_true[j] = total_context[j] / total
        ground_truth[i] = y_true
    return ground_truth

def sort_predictions(pred_over_time):
    size = len(pred_over_time[0])
    for i, pred in enumerate(pred_over_time):
        pred = pred_over_time[i]
        pred_over_time[i] = dict()
        for j in range(size):
            pred_over_time[i][j] = pred[j]
        pred_over_time[i] = sorted(pred_over_time[i].items(), key=lambda item: item[1], reverse=True)
    
    for i in range(len(pred_over_time)):
        for j in range(size):
            pred_over_time[i][j] = pred_over_time[i][j][1]

'''
def target_to_total_context(training_data):
    target_total_context = defaultdict(float)  # Use defaultdict for cleaner initialization
    num_appearances = defaultdict(int)  # Use defaultdict for cleaner initialization
    
    for target, context in training_data:
        target_idx = target[1]
        num_appearances[target_idx] += 1
        for context_word in context:
            target_total_context[target_idx] += context_word[0]
    
    return target_total_context, num_appearances

def ground_truth(vocab_size, training_data, window_size):
    target_total_context, num_appearances = target_to_total_context(training_data)
    ground_truth = {}
    
    for i in range(vocab_size):
        total_context = target_total_context[i]
        total = 2 * num_appearances[i] * window_size
        y_true = {j: total_context[j] / total for j in range(len(total_context))}
        ground_truth[i] = y_true
    
    return ground_truth

def sort_predictions(pred_over_time):
    size = len(pred_over_time[0])
    sorted_pred_over_time = []
    
    for pred in pred_over_time:
        sorted_pred = sorted(pred.items(), key=lambda item: item[1], reverse=True)
        sorted_pred_over_time.append([item[1] for item in sorted_pred])
    
    return sorted_pred_over_time
'''