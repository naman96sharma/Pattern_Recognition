# Machine Learning Algorithms from Scratch
This repository builds some common Machine Learning algorithms from scratch on Python. The aim is to improve an understanding of how these algorithms work by implementing them without using the easily available `scikit-learn`. For this purpose, I use two small use cases: spam filtering and hand digit recognition. At the end of every jupyter notebook, a code for the ML class is provided for your own experimentation.

## Algorithms Implemented
1. **Beta Binomial Naive Bayes**: [Naive Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) is a probabilistic classifier which uses the Bayes' theorem. The version implemented here uses a beta prior distribution which is the conjugate of a binomial distribution.
2. **Gaussian Naive Bayes**: Another implementation of Naive Bayes, but this time the data is assumed to be distributed based on a Gaussian distribution instead.
3. **Logistic Regression**: Common classification algorithm used for cases when the data belongs to one of two classes.
4. **K Nearest Neighbors**: Non parametric method for classification/regression.
5. **Principle Component Analysis**: [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) is an unsupervised learning algorithm used for dimensionality reduction of data and visualizing your data in a 2D space.
6. **Linear Discriminant Analysis**: [LDA](https://en.wikipedia.org/wiki/Linear_discriminant_analysis) is a supervised learning algorithm that can also be used for dimensionality reduction, feature extraction and classification.
7. **Support Vector Machine**: [SVM](https://en.wikipedia.org/wiki/Support-vector_machine) is one of the most commonly used algorithms before neural networks became feasible. They are still commonly used in cases when the amount of training data available is limited. It can be used for both classification and regression. For SVM, I use the implementation provided by `scikit-learn`, with the aim of looking at the performance of the classifier instead.
8. **Multi Layer Perceptron**: [MLP](https://en.wikipedia.org/wiki/Multilayer_perceptron) is a type of artificial neural network with multiple hidden connected layers. The 2-layer MLP is implemented in `network.py`, and used in the `MLP.ipynb` Jupyter notebook. This notebook also uses the `check_gradients.py` to ensure that the implemented MLP gives the expected results.
9. **Convolutional Neural Network**: CNN implementation is done using Keras as the backend.

## Running the code
The algorithms are implemented with Python on Jupyter notebooks, allowing easier viewing. The first 4 algorithms use the spam dataset provided in the `spamData.mat` file. The next 5 algorithms use the hand written digits dataset. The hand written digits dataset is automatically downloaded while running the Jupyter notebooks.