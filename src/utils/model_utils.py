
import numpy as np

class SkipGramModel():
    def __init__(self, vocab_size, embedding_dim, target_id=0, learning_rate=0.001):
        self.embedding_matrix = np.random.uniform(-1, 1, (vocab_size, embedding_dim))
        self.weights = np.random.uniform(-1, 1, (embedding_dim, vocab_size))

        self.learning_rate = learning_rate

        self.target_id=target_id

        self.vectors_over_time = []
        self.predictions_over_time = []
        self.loss_curve = []
    
    @staticmethod
    def softmax(score):
        exp_scores = np.exp(score - np.max(score))
        return exp_scores / exp_scores.sum()
    
    def loss_fn(self, y_pred, context):
        epsilon = 1e-8
        loss = -np.sum(context * np.log(np.maximum(y_pred, epsilon)), axis=-1).mean()

        return loss
    
    # for predictions, we perform an inner product with the weights, then softmax the score to get a distribution we compare to ground truth context
    def predict(self, input_id):
        try:    
            score = np.dot(self.embedding_matrix[input_id], self.weights)
            y_pred = SkipGramModel.softmax(score)
        except:
            print(input_id)
            y_pred = np.zeros(self.embedding_matrix[input_id].shape)
       
        return y_pred

    # backward pass transfers the gradient from the prediction error (y_pred - context)
    # to the weights by performing an outer product with the current word embedding. 
    # this is then used to update the embedding vector by performing an inner product on the new weights with the error
    def backward(self, input_id, context, y_pred):
        for context_vector in context:
            # Calculate the error as the difference between the prediction and the true context word
            error = y_pred - context_vector
            
            # Update the output vectors (model weights)
            self.weights -= self.learning_rate * np.outer(self.embedding_matrix[input_id], error)

            # Update the input vectors (embedding vector)
            self.embedding_matrix[input_id] -= self.learning_rate * np.dot(self.weights, error)

    # updated when the model runs to get visualizations
    def update_model_states(self, word_vectors_2d=None, predictions=None, loss=None):
        if word_vectors_2d is not None:
            self.vectors_over_time.append(word_vectors_2d)
        if predictions is not None:
            self.predictions_over_time.append(predictions)
        if loss is not None:
            self.loss_curve.append(loss)

class Tokenizer():
    def __init__(self):
        self.vocab = []
        self.index_to_key = {}
        self.key_to_index = {}
    
    def fit(self, words):
        # get unique words
        vocab = set(words)
        # get index for each unique word and create forward and backward map
        self.index_to_key = {i : word for i, word in enumerate(vocab)}
        self.key_to_index = {word : i for i, word in enumerate(vocab)}
        self.vocab = list(vocab)