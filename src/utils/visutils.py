import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import TruncatedSVD, PCA
import numpy as np

# getting colors for our vectors, one per vector so we can easily distinguish the points
def generate_unique_colors(num_colors):
    cmap = plt.get_cmap('tab20')
    return cmap(np.linspace(0, 1, num_colors))

def visualize_vectors_over_time(vectors_over_time, index_to_key, num_words=20):
    plt.ioff()

    bound = 2
    vectors_to_display = []
    # Only select vectors for the first num_words words
    for vectors in vectors_over_time:
        vectors_shown = [v for i, v in enumerate(vectors[:num_words]) if index_to_key[i] != 'voices']
        vectors_to_display.append(vectors_shown)

    # Generate unique colors for each word
    num_colors = num_words
    unique_colors = generate_unique_colors(num_colors)

    # Create a figure and axis
    # Initialize scatter plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sc = ax.scatter([], [])

    # Set the axis limits
    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)

    # Create a list to hold the annotations
    annotations = [ax.annotate('', (0, 0)) for _ in range(num_words)]

    def update(frame):
        # Update the scatter plot data for each frame
        sc.set_offsets(vectors_to_display[frame])
        sc.set_color(unique_colors[:len(vectors_to_display[frame])])  # Set colors
        
        # Update annotations with word labels
        for i, (x, y) in enumerate(vectors_to_display[frame]):
            annotations[i].set_text(index_to_key[i])  # Set word label
            annotations[i].set_position((x, y))  # Set position
            annotations[i].set_visible(True)  # Make annotation visible

        return sc, *annotations

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(vectors_to_display), blit=True)

    plt.ion()
    
    return ani


def get_context_distribution(dataset, input_id, vocab_size):
    context_accumulator = np.zeros(vocab_size)

    # Iterate through the corpus data
    for sample in dataset:
        if sample['input_id'] == input_id:
            context_accumulator += np.sum(sample['context'], axis=0)

    # Calculate the distribution by normalizing the accumulator
    context_distribution = context_accumulator / context_accumulator.sum()

    return context_distribution


def visualize_predictions_over_time(pred_over_time, y_true, index_to_key, num_words=20):
    plt.ioff()

    # Create a figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    lines, = ax.plot([], [], color='red', label='Predicted')
    
    ax.plot(y_true, color='green', label='Ground Truth')

    # Set the axis limits
    ax.set_xlim(0, len(index_to_key))
    ax.set_ylim(0, 1)  # Assuming context probabilities are between 0 and 1
    
    # Set labels and legend
    ax.set_xlabel('Vocabulary')
    ax.set_ylabel('Probability')
    ax.set_title('Predicted vs Ground Truth Probability for "treasure"')
    ax.legend(loc='upper right')

    selected_indices = list(np.where(y_true > 0)[0])
    selected_labels = [index_to_key[i] for i in selected_indices]

    ax.set_xticks(selected_indices)
    ax.set_xticklabels(selected_labels, rotation=90)

    def update(frame):
        prob_list = pred_over_time[frame]
        lines.set_data(range(len(prob_list)), prob_list)

        return lines,

    # Create the animation
    ani = FuncAnimation(fig, update, frames=len(pred_over_time), blit=True)

    plt.tight_layout()

    plt.ion()

    return ani