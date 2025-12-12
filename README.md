# Fall25 ML Final Project

**Team Members**
Zhongbo Liu (zl6386)
Yufei Liu (yl14412)


## Problem Formulation

In many real applications—such as online shopping—users want to upload a picture and find items that look similar.  
For example, if a user uploads a boot, the system should return boots, not shirts.

However, comparing images directly at the pixel level does not work. Two boots may have different lighting or angle, while two different clothing types may share similar pixel patterns.  
Therefore, we need a better way to represent images.

This project asks a simple question:

**Can we learn a feature representation where similar fashion items are close together and different items are far apart?**

To explore this question, we build a full representation-learning pipeline:

- Train a CNN to learn meaningful features from FashionMNIST  
- Extract embeddings from the CNN’s hidden layer  
- Visualize the feature space using PCA and t-SNE  
- Cluster the embeddings using KMeans  
- Build a similarity search system that retrieves the most similar images based on cosine similarity  

Our goal is to examine:

- whether the CNN learns useful visual patterns  
- how different fashion categories are organized in feature space  
- whether the learned embeddings can support real applications such as clustering and search-by-image  

Overall, this project shows how deep learning can be used not only for classification, but also for understanding image structure and building practical tools.

---

## Approach

To learn a useful visual representation of FashionMNIST, we design a complete pipeline that goes beyond simple classification.  
Our approach contains five main stages.

### (1) CNN Model for Feature Learning

We first train a Convolutional Neural Network (CNN) on FashionMNIST.  
The CNN is responsible for learning high-level visual features, such as shape, edges, and category cues.

The network has two convolutional blocks followed by fully-connected layers.  
Instead of only using the final prediction, we extract the **128-dimensional embedding** from the second-to-last layer.  
This embedding becomes the *feature vector* for each image and serves as the foundation for all later analysis.

### (2) PCA for Dimensionality Reduction & Reconstruction

Next, we apply Principal Component Analysis (PCA) for two purposes.

**PCA-2 Visualization**

- We reduce raw pixel data (784 → 2) to observe how images distribute in a low-dimensional space  
- This helps compare raw features versus CNN features  

**PCA-50 Reconstruction**

- We keep 50 principal components and reconstruct the images  
- This shows how much visual information is preserved when using a compressed representation  

PCA helps us understand what structures exist in the data even before deep learning.

### (3) t-SNE for Non-Linear Embedding Visualization

To explore the deeper structure of the learned representation, we apply t-SNE on the CNN embeddings.

- 128-dimensional CNN features are mapped to 2D  
- Each point is colored by its true label  
- Well-formed clusters indicate that the CNN has learned meaningful semantic structure  

t-SNE is particularly powerful for revealing non-linear separations between categories.

### (4) KMeans Clustering for Unsupervised Structure Discovery

We apply KMeans (k = 10) to the PCA-50 features to check whether similar items group naturally without using labels.

- If clusters align with real classes, the representation is meaningful  
- If clusters mix, it indicates harder categories (e.g., shirt vs T-shirt)  

We visualize KMeans clusters again in PCA-2 space, showing how well the representation organizes the dataset without supervision.

### (5) Similarity Search Using Cosine Similarity

Finally, we build a simple but highly practical image retrieval system.

Steps include:

- Normalize all CNN embeddings  
- Compute cosine similarity between a query vector and all test vectors  
- Retrieve the Top-K nearest neighbors  
- Display query versus similar items  

This demonstrates a real application of representation learning:  
**search-by-image**, which is widely used in fashion apps, e-commerce, and content recommendation systems.

---

## Evaluation

We evaluate our system from multiple perspectives, including classification performance, learned feature representation quality, clustering behavior, and similarity retrieval results.

### (1) CNN Training Performance

The CNN reaches **92% test accuracy**, significantly higher than a simple MLP baseline.  
This indicates that convolutional features capture far more meaningful visual patterns than raw pixels.

- **Loss Curve**: Train loss steadily decreases → model learns effectively  
- **Test Loss** stays close → model is not overfitting  
- **Accuracy Curve**: Train and test accuracy move together → stable generalization  

### (2) Confusion Matrix

The confusion matrix highlights how the model behaves on each class.

- Classes like **boots, sneakers, and bags** are recognized very well  
- Ambiguous classes such as **shirt vs T-shirt** and **pullover vs coat** show more confusion  

This indicates that remaining errors mostly come from **intrinsic category similarity**, not model weakness.

### (3) PCA Analysis

#### (i) PCA-2 on Raw Pixels

When applying PCA directly to raw images:

- Different categories are heavily mixed  
- The structure is not clearly separable  
- Raw pixel space does not encode meaningful fashion semantics  

This provides a baseline to compare with CNN embeddings later.

#### (ii) PCA Reconstruction (50 Components)

By reconstructing images using only 50 principal components, we observe:

- Global shape is preserved  
- Fine details are lost  

This demonstrates how PCA compresses information and which visual components are most important.

### (4) KMeans Clustering (Unsupervised)

Using PCA-50 features, KMeans forms clusters that roughly match real classes:

- Footwear categories cluster relatively well  
- Tops (shirt / pullover / coat) mix together, showing visual similarity  

This confirms that even without labels, the feature space contains meaningful structure.

When visualized in PCA-2 space, clusters appear as distinct regions, demonstrating the unsupervised organization of the dataset.

### (5) t-SNE Visualization of CNN Embeddings

t-SNE on the 128-dimensional CNN embeddings reveals:

- Clear, well-separated clusters for many classes  
- Shoes, bags, and trousers form extremely distinct groups  
- Ambiguous clothing classes overlap but still show partial separation  

Compared with PCA on raw pixels, this visualization clearly proves that:

**CNN embeddings contain rich semantic structure that raw pixels cannot provide.**

### (6) Similarity Search Results (Search-by-Image)

This is the strongest evidence of a meaningful representation.

Given a query image:

- The top-5 retrieved items almost always share the same category  
- Even when labels differ slightly, the retrieved items look visually similar  
- Footwear retrieves footwear, pants retrieve pants, coats retrieve coats  
- The system works purely based on deep features and cosine similarity (no labels)  

This shows that the learned embedding space naturally supports real-world applications such as:

- Fashion image search  
- Recommendation systems  
- Content-based retrieval  

Overall, the learned CNN embedding space captures meaningful semantic structure, clearly outperforming raw pixel representations and supporting clustering, visualization, and similarity-based retrieval.

---

## Presentation

This project is presented as a clean and easy-to-navigate GitHub repository.  
All code, results, and explanations are organized clearly so that the workflow and conclusions can be easily understood and reproduced.

### (1) Repository Structure

Fall25-ML-Project/
│── results/ # All generated figures and visual results
│── ML final project.ipynb # Complete implementation and experiments
│── README.md # Project report and explanations


All model training, feature extraction, visualization, and analysis are implemented in a single Jupyter notebook.  
This unified structure allows readers to follow the full pipeline without switching between multiple scripts.

### (2) Clarity & Visualization

The `results/` folder contains all experimental outputs, including:

- Training and test accuracy / loss curves  
- Confusion matrix  
- PCA visualizations and reconstruction results  
- t-SNE plots  
- KMeans clustering results  
- Similarity search examples  

Each figure directly corresponds to a section in the notebook and the evaluation discussion in this README.  
Visual results are clearly labeled and organized, allowing readers to quickly verify and interpret the findings.

### (3) Reproducibility

All experiments can be reproduced by running the Jupyter notebook from top to bottom.  
The notebook includes data loading, model training, feature extraction, visualization, and evaluation in a single, consistent workflow.

This design ensures that all reported figures and results are fully reproducible from the provided code.

---

## Bonus

In addition to the required components, this project includes several advanced techniques that go beyond the standard course material.  
These additions demonstrate a deeper understanding of representation learning and practical, real-world applications.

### (1) Non-Linear Embedding with t-SNE

While PCA provides linear dimensionality reduction, t-SNE is used to explore the non-linear structure of CNN embeddings.  
It reveals clear and meaningful clusters that are not visible in raw pixel space.

### (2) Image Reconstruction with PCA (50 Components)

PCA-based reconstruction is used to analyze how much visual information is preserved using only 50 dimensions.  
This illustrates the trade-off between compression and image quality.

### (3) Search-by-Image System Using Cosine Similarity

A functional similarity search system is implemented.

- Given a query image, the system retrieves the top-K most similar images  
- Similarity is computed using CNN embeddings and cosine similarity  

This demonstrates that the learned representation enables:

- Visual search  
- Recommendation systems  
- Content-based retrieval  

### (4) Analysis of Learned Representations

We analyze how the CNN organizes classes in feature space by comparing:

- PCA of raw pixels  
- PCA of embeddings  
- t-SNE of embeddings  
- KMeans clustering results  

This multi-angle analysis provides insight into what the network learns and how representation quality affects downstream tasks.
