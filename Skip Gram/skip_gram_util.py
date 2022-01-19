import re
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as ani
from random import random

# process contents of a text file into list of words
def get_data(filename):
    contents = []
    with open(filename) as f:
        contents = f.read()
    contents = contents.split('.')
    data = []
    for line in contents:
        words = re.findall("\w+", line)
        for w in words:
            if len(w) > 0:
                data.append(w.lower())
    return data, contents

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
    from skip_gram import forward_pass, cos_similarity_dict
    num_similar = 10
    for word in testset:
        h, y, s = forward_pass(word_to_int[word], embedding_matrix, weights_out)
        similarities_in, similarities_out = cos_similarity_dict(word, embedding_matrix, weights_out.T, word_to_int, int_to_word)
        # print target word, context prediction with certainty, and a few of the most similar word vectors 
        print('{:<15}'.format('Target:'), word)
        print('{:<15}'.format('Prediction:'), int_to_word[np.argmax(y)])
        print('{:<15}'.format('Probability:'), y[np.argmax(y)].round(4))
        print('{:<15}'.format('Most similar Input:'), list(similarities_in)[:num_similar])
        print('{:<15}'.format('Most similar Output:'), list(similarities_out)[:num_similar], '\n')

# visualize data as vectors (mode = 0) or predictions (mode != 0) over time
def visualize_data(vectors_over_time, x, y_true, pred_over_time, epochs, int_to_word, mode = 0):
    fig = plt.figure()
    colors = [[random(), random(), random()] for _ in range(len(x))]
    def visualize_data_1(i=int):
        if i < len(vectors_over_time):
            fig.clear()
            plt.scatter(vectors_over_time[i][0], vectors_over_time[i][1], c=colors)
            for j in range(len(vectors_over_time[0][0])):
                plt.annotate(int_to_word[j], xy=(vectors_over_time[i][0][j], vectors_over_time[i][1][j]), xytext=(0,7), textcoords='offset points', ha='left', va='center')

    def visualize_data_2(i=int):
        if i < len(pred_over_time):
            fig.clear()
            plt.plot(x, y_true, color='green')
            plt.plot(x, pred_over_time[i], color='red')
    if mode == 0:
        animation = ani.FuncAnimation(fig, visualize_data_1, interval=epochs/10)
    else:
        animation = ani.FuncAnimation(fig, visualize_data_2, interval=epochs/10)
    plt.show()

def visualize_data_discrete(vectors_over_time, x, epochs, int_to_word):
    import matplotlib
    import tkinter as Tk
    from matplotlib.widgets import Slider
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    matplotlib.use('TkAgg')
    root = Tk.Tk()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.subplots_adjust(bottom=0.1)

    canvas = FigureCanvasTkAgg(fig, root)
    canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)
    
    ax_time = fig.add_axes([0.12, 0.95, 0.78, 0.03])
    slider = Slider(ax_time, 'Epoch', 0, epochs/10, valinit=0, valstep=1, dragging=True)

    colors = [[random(), random(), random()] for _ in range(len(x))]

    def visualize_data_1(i):
        if i < len(vectors_over_time):
            ax.clear()
            ax.scatter(vectors_over_time[i][0], vectors_over_time[i][1], c=colors)
            for j in range(len(vectors_over_time[0][0])):
                ax.annotate(  int_to_word[j], 
                            xy=(vectors_over_time[i][0][j], vectors_over_time[i][1][j]), 
                            xytext=(0,7), textcoords='offset points', 
                            ha='left', va='center', c=colors[j])
            fig.canvas.draw_idle()

    def visualize_data_2(i):
        if i < len(pred_over_time):
            ax.clear()
            ax.plot(x, y_true, color='green')
            ax.plot(x, pred_over_time[i], color='red')

    slider.on_changed(visualize_data_1)
    Tk.mainloop()




