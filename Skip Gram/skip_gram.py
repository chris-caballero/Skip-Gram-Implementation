import numpy as np
from math import exp

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

# evaluates the loss function for the predicted score
def loss(context, score):
    loss = len(context)*np.log(np.sum(np.exp(score)))
    for context_word in context:
        loss -= score[context_word[1]]
    return loss

# generate prediction of context on current target word
def forward_pass(target_idx, embedding_matrix, weight_matrix_out):
    # choosing the right word vector from out embedding matrix
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
    error = len(context)*y_pred.copy()
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
    from sklearn.decomposition import TruncatedSVD
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
    if mode == 1:
        similarities = dict()
        word_vector = embedding_matrix[word_to_int[word]]
        for i in range(len(embedding_matrix)):
            similarities[int_to_word[i]] = cos_dist(word_vector, embedding_matrix[i]).round(4)
        return sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    elif mode == 2:
        similarities = dict()
        word_vector = embedding_matrix[word_to_int[word]]
        for i in range(len(weights_out)):
            similarities[int_to_word[i]] = cos_dist(word_vector, weights_out[i]).round(4)
        return sorted(similarities.items(), key=lambda item: item[1], reverse=True)

    similarities_in = dict()
    similarities_out = dict()
    word_vector = embedding_matrix[word_to_int[word]]
    for i in range(len(embedding_matrix)):
        similarities_in[int_to_word[i]] = cos_dist(word_vector, embedding_matrix[i]).round(4)
        similarities_out[int_to_word[i]] = cos_dist(word_vector, weights_out[i]).round(4)
    return sorted(similarities_in.items(), key=lambda item: item[1], reverse=True), sorted(similarities_out.items(), key=lambda item: item[1], reverse=True)

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