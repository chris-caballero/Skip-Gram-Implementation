## **Skip Gram Implementation**

Undertaking this project as one of my initial explorations into machine learning was an invaluable experience, greatly enriching my understanding of neural network development. Through this endeavor, I have immersed myself in the intricacies of creating neural networks, allowing me to grasp the underlying concepts with greater depth. Overall, this project played a huge role in expanding my knowledge and developing a deeper appreciation for the nuances involved in neural network development.

***

### **How to Run**

To run the model on the toy corpus, follow these steps:

1. Execute the script `scripts/skip_gram_test.py`.
2. Specify the toy corpus file `data/word_dataset.txt`.

```shell
python skip_gram_test.py
```

This will generate **.gif** files in the `visualizations` folder.

***

### **Results**

In order to verify the results, I tracked the vectors during the training process and observed the convergence of probabilities. The implementation relies on a straightforward Python and NumPy approach, which inherently limits its scalability. However, it performs effectively at smaller scales, fulfilling its intended purpose and providing an interactive and engaging experience.

#### Vector Movement Animation

The animation below visualizes the convergence of word vectors based on the word distributions in the toy corpus. As the training progresses, various semantic relationships start to emerge.

<img src="visualizations/vector_movement.gif" alt="Vector Movement Animation" width="400">

#### Probability Convergence Animation

The animation below demonstrates the model converging to the ground truth distribution of words in the context of **'every'** within the dataset. The probabilities are sorted for comparison.

<img src="visualizations/probability_convergence.gif" alt="Probability Convergence Animation" width="400">

