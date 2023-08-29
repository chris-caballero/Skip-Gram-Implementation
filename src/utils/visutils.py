import matplotlib.pyplot as plt
import matplotlib.animation as ani
from random import random


# visualize data as vectors (mode = 0) or predictions (mode != 0) over time
def visualize_data(vectors_over_time, x, y_true, pred_over_time, epochs, int_to_word, test_word=None, mode=0):
    fig = plt.figure()
    animation = None

    colors = [[random(), random(), random()] for _ in range(len(x))]

    def visualize_data_1(i):
        fig.clear()
        plt.scatter(vectors_over_time[i][0], vectors_over_time[i][1], c=colors)
        for j in range(len(vectors_over_time[0][0])):
            plt.annotate(int_to_word[j], xy=(vectors_over_time[i][0][j], vectors_over_time[i][1][j]), xytext=(
                0, 7), textcoords='offset points', ha='left', va='center')
        plt.title('Word Vector Movement in Training')

    def visualize_data_2(i):
        fig.clear()
        plt.plot(x, y_true, color='green', label='Ground Truth')
        plt.plot(x, pred_over_time[i], color='red', label='Predicted')
        plt.title(f'Predicted Context of \'{test_word}\' vs Ground Truth')
        plt.legend(loc='upper right')
        plt.ylabel('Probability')

    if mode == 0:
        animation = ani.FuncAnimation(fig, visualize_data_1, frames=len(vectors_over_time), interval=epochs/10)
        
    else:
        animation = ani.FuncAnimation(fig, visualize_data_2, frames=len(pred_over_time), interval=epochs/10)
        
    plt.show(block=False)

    return animation


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
    slider = Slider(ax_time, 'Epoch', 0, epochs/10,
                    valinit=0, valstep=1, dragging=True)

    colors = [[random(), random(), random()] for _ in range(len(x))]

    def visualize_data_1(i):
        ax.clear()
        ax.scatter(vectors_over_time[i][0], vectors_over_time[i][1], c=colors)
        for j in range(len(vectors_over_time[0][0])):
            ax.annotate(int_to_word[j], xy=(vectors_over_time[i][0][j], vectors_over_time[i][1][j]), xytext=(
                0, 7), textcoords='offset points', ha='left', va='center', c=colors[j])
        fig.canvas.draw_idle()

    slider.on_changed(visualize_data_1)
    Tk.mainloop()
