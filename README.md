# Fall25 ML Final Project

**Team Members**
Zhongbo Liu (zl6386)
Yufei Liu (yl14412)


This project studies representation learning on the FashionMNIST dataset.The goal is to learn meaningful feature embeddings so that visually similar fashion items are close in feature space while different items are far apart.

We first train a Convolutional Neural Network (CNN) to extract high-level visual features. Embeddings from the hidden layer are then used for further analysis. Principal Component Analysis (PCA) is applied for dimensionality reduction and image reconstruction, while t-SNE is used to visualize the non-linear structure of the learned feature space. KMeans clustering is performed to study unsupervised grouping of fashion categories.

Finally, we implement a similarity search system based on cosine similarity. Given a query image, the system retrieves the top-5 most visually similar images using CNN embeddings. This demonstrates how representation learning can support real-world applications such as search-by-image and recommendation.

All experiments, analysis, and implementations are contained in the Jupyter notebook. Result figures and visualizations are provided in the `results` folder.
