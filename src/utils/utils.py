import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def get_data(filename):
    contents = []
    with open(filename, 'r') as f:
        lines = f.read()
    
    return lines

def process_data(text):
    stop_words = set(stopwords.words('english'))

    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    cleaned_text = text.strip()

    words = cleaned_text.split(' ')
    stop_words_removed = [word.lower() for word in words if word.lower() not in stop_words]

    return stop_words_removed

def display_text(text, words=None):
    print('-'*6 + f'\nCORPUS\n' + '-'*6)
    for i, line in enumerate(corpus.split('.')):
        print(line)

        if i > 5:
            break

    if words is not None:
        print()
        print('-'*5 + f'\nWORDS\n' + '-'*5)
        print(words[:10])

# Tokenizer class is very simple, just getting the encodings for the words and maintaining the vocabulary
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

def encode_words(words, tokenizer):
    # encode the words as their indices for training
    encoding = [tokenizer.key_to_index[word] for word in words]

    return encoding


import numpy as np

def one_hot(context_id, vocab_size):
    one_hot_vector = np.zeros(vocab_size)
    one_hot_vector[context_id] = 1.0
    
    return one_hot_vector

def create_dataset(encoding, tokenizer, window_size=2):
    dataset = []
    for i in range(len(encoding)):
        start = max(0, i - window_size)
        end = min(len(encoding), i + window_size + 1)

        context = encoding[start:i]
        context.extend(encoding[i+1:end])

        # target = one_hot(context, tokenizer)
        
        dataset.append({
            'input_id': encoding[i], 
            'context': [one_hot(context_id, len(tokenizer.vocab)) for context_id in context]
        })

    return dataset



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
        



def train(model, training_data, epochs=100, verbose=True, reducer=None):
    for epoch in range(epochs+1):
        for sample in training_data:
            # get our input word and its current context
            input_id = sample['input_id']
            context = np.array(sample['context'])
            
            # predict the context and get the loss
            y_pred = model.predict(input_id)
            loss = model.loss_fn(y_pred, context)

            # update the weights and embeddings
            model.backward(input_id, context, y_pred)

        # every so often we add to our information about the model
        if reducer and epoch % 10 == 0:
            x = reducer.transform(model.embedding_matrix)
            model.update_model_states(word_vectors_2d=x)
        if epoch % 50 == 0 and verbose:
            print(f"Epoch: {epoch} - Loss: {loss}")
        
        model.update_model_states(predictions=model.predict(model.target_id), loss=loss)

def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity

def similar_words(model, target_id=0, vector=None, top_k=5):
    # Assuming you have a list of words and their corresponding embedding vectors
    embedding_matrix = model.embedding_matrix  # Your embedding matrix

    # Get the embedding vector of the target word
    if vector is None:
        target = embedding_matrix[target_id]
    else:
        target = vector

    # Calculate cosine similarity between the target word and all other words
    similarities = {}
    for index in tokenizer.index_to_key:
        if index != target_id:
            similarity = cosine_similarity(target, embedding_matrix[index])
            similarities[tokenizer.index_to_key[index]] = similarity

    # Sort the similar words by their cosine similarity
    similar_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    # Print the top similar words
    if vector is None:
        print("Similar words for", tokenizer.index_to_key[target_id])
    else:
        print("Similar words")
    for word, similarity in similar_words[:top_k]:
        print(word, ":", similarity)

    return similar_words