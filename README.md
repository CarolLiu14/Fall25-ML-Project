# Fall25 ML Final Project

**Team Members**
Zhongbo Liu (zl6386)
Yufei Liu (yl14412)


**Problem Formulation**
In many real applications—such as online shopping—users want to upload a picture and find items that look similar. For example, if a user uploads a boot, the system should return boots, not shirts. However, comparing images directly at the pixel level does not work. Two boots may have different lighting or angle, while two different clothing types may share similar pixel patterns. So we need a better way to represent images.
This project asks a simple question:
Can we learn a feature representation where similar fashion items are close together and different items are far apart?
To explore this, we build a full representation-learning pipeline:
*Train a CNN to learn meaningful features from FashionMNIST
*Extract embeddings from the CNN’s hidden layer
*Visualize the feature space using PCA and t-SNE
*Cluster the embeddings using KMeans
Build a similarity search system that retrieves the most similar images based on cosine similarity
Our goal is to see:
*whether the CNN learns useful visual patterns
*how different fashion categories are organized in feature space
*whether the learned embeddings can support real applications such as clustering and search-by-image
Overall, this project shows how deep learning can be used not only for classification, but also for understanding image structure and building practical tools.

**Approach**
To learn a useful visual representation of FashionMNIST, we design a complete pipeline that goes beyond simple classification. Our approach contains five main stages:
(1) CNN Model for Feature Learning
We first train a Convolutional Neural Network (CNN) on FashionMNIST.
The CNN is responsible for learning high-level visual features, such as shape, edges, and category cues.
The network has two convolutional blocks followed by fully-connected layers.
Instead of only using the final prediction, we extract the 128-dimensional embedding from the second-to-last layer.
This embedding becomes the “feature vector” for each image.
These embeddings serve as the foundation for all later analysis.
(2) PCA for Dimensionality Reduction & Reconstruction
Next, we apply Principal Component Analysis (PCA) for two purposes:
a. PCA-2 Visualization
We reduce raw pixel data (784 → 2) to observe how images distribute in a low-dimensional space.
This helps compare raw features vs CNN features.
b. PCA-50 Reconstruction
We keep 50 principal components and reconstruct the images.
This shows how much visual information is kept when using a compressed representation.
PCA helps us understand what structures exist in the data even before deep learning.
(3) t-SNE for Non-Linear Embedding Visualization
To explore the deeper structure of the learned representation, we apply t-SNE on the CNN embeddings.
We map 128-dimensional CNN features to 2D
Each point is colored by its true label
Well-formed clusters indicate that the CNN has learned meaningful semantic structure
t-SNE is particularly powerful for revealing non-linear separations between categories.
(4) KMeans Clustering for Unsupervised Structure Discovery
We apply KMeans (k = 10) to the PCA-50 features to check whether similar items group naturally without using labels.
If clusters align with real classes, it means the representation is meaningful
If they mix, it indicates harder categories (e.g., shirt vs T-shirt)
We visualize KMeans clusters again in PCA-2 space.
This step shows how well the representation organizes the dataset without supervision.
(5) Similarity Search Using Cosine Similarity
Finally, we build a simple but highly practical image retrieval system.
Steps:
Normalize all CNN embeddings
Compute cosine similarity between a query vector and all test vectors
Retrieve the Top-K nearest neighbors
Display query vs similar items
This demonstrates a real application of representation learning:
“search by image”, widely used in fashion apps, e-commerce, and content recommendation.

**Evaluation**

We evaluate our system from multiple perspectives:
classification performance, learned feature representation quality, clustering behavior, and similarity retrieval results.

(1) CNN Training Performance
The CNN reaches 92% test accuracy, significantly higher than a simple MLP baseline.
This indicates that convolutional features capture far more meaningful visual patterns than raw pixels.
Loss Curve：Train loss steadily decreases → model learns effectively
Test Loss stays close → model is not overfitting
Accuracy Curve：Train and test accuracy move together → stable generalization
(2) Confusion Matrix
The confusion matrix highlights how the model behaves on each class.
Classes like boots, sneakers, bags are recognized very well, and ambiguous classes such as shirt vs T-shirt, pullover vs coat show more confusion
This indicates that remaining errors mostly come from intrinsic category similarity, not model weakness.
(3) PCA Analysis
(i) PCA-2 on Raw Pixels
When applying PCA directly to raw images:
* Different categories are heavily mixed
* The structure is not clearly separable
* Raw pixel space does not encode meaningful fashion semantics
This provides a baseline to compare with CNN embeddings later.
(ii) PCA Reconstruction (50 Components)
By reconstructing images using only 50 principal components, we see:
* Global shape is preserved
* Fine details are lost
This demonstrates how PCA compresses information and which parts are visually most important.
(4) KMeans Clustering (Unsupervised)
Using PCA-50 features, KMeans forms clusters that roughly match real classes:
* Footwear categories cluster relatively well
* Tops (shirt/pullover/coat) mix together, showing visual similarity
This confirms that even without labels, the feature space contains meaningful structure.
When visualized in PCA-2, clusters appear as distinct regions—showing the unsupervised organization of the dataset.
(5) t-SNE Visualization of CNN Embeddings
t-SNE on the 128-dim CNN embeddings reveals:
* Clear, well-separated clusters for many classes
* Shoes, bags, trousers form extremely distinct groups
* Ambiguous clothing classes overlap but still show partial separation
Compared with PCA on raw pixels, this visualization clearly proves:
CNN embeddings contain rich semantic structure that raw pixels cannot provide.
(6) Similarity Search Results (Search-by-Image)
This is the strongest evidence of a meaningful representation.
Given a query image:
* The top-5 retrieved items almost always share the same category
* Even when labels differ slightly, the retrieved items look visually similar
* Footwear items retrieve footwear, pants retrieve pants, coats retrieve coats
* The system works purely based on deep features + cosine similarity (no labels)
This shows that our deep embedding space naturally supports real-world applications such as:
* fashion image search
* recommendation systems
* content-based retrieval
This final step demonstrates the practical value of representation learning beyond classification.

Overall, the learned CNN embedding space captures meaningful semantic structure, clearly outperforming raw pixel representations and supporting clustering, visualization, and similarity-based retrieval.

**Presentation**

This project is presented as a clean and easy-to-navigate GitHub repository. All code, results, and explanations are organized clearly so that the workflow and conclusions can be easily understood and reproduced.

(1) Repository Structure
Fall25-ML-Project/
│── results/                   # All generated figures and visual results
│── ML final project.ipynb     # Complete implementation and experiments
│── README.md                  # Project report and explanations
All model training, feature extraction, visualization, and analysis are implemented in a single Jupyter notebook. This unified structure allows readers to follow the full pipeline without switching between multiple scripts.
(2) Clarity & Visualization
The results/ folder contains all experimental outputs, including:
Training and test accuracy/loss curves, confusion matrix, PCA visualizations, reconstruction results, t-SNE plots, KMeans clustering results, and similarity search examples.
Each figure directly corresponds to a section in the notebook and the evaluation discussion in this README.
Visual results are clearly labeled and organized, allowing readers to quickly verify and interpret the experimental findings.
(3) Reproducibility
All experiments can be reproduced by running the Jupyter notebook from top to bottom. The notebook includes data loading, model training, feature extraction, visualization, and evaluation in a single, consistent workflow.This design ensures that all reported figures and results are fully reproducible from the provided code.

**Bonus**

In addition to the required components, this project includes several advanced techniques that go beyond the standard course material. These additions highlight a deeper understanding of representation learning and demonstrate practical, real-world applications.

(1) Non-Linear Embedding with t-SNE
While PCA provides linear dimensionality reduction, we further use t-SNE to explore the non-linear structure of CNN embeddings.
t-SNE reveals clear and meaningful clusters that are not visible in raw pixel space, providing strong evidence of the network’s ability to organize high-level visual patterns.
(2) Image Reconstruction with PCA (50 Components)
We perform PCA-based reconstruction to analyze how much visual information is preserved using only 50 dimensions.
This experiment illustrates the trade-off between compression and image quality, and helps interpret the role of principal components in representing fashion images.
(3) Search-by-Image System Using Cosine Similarity
One of the most practical contributions is a functional similarity search system.
Given a query image, the system retrieves the top-K most similar images based solely on CNN embeddings and cosine similarity.
This demonstrates that the learned representation is not only useful for classification, but directly enables:
Visual search
Recommendation systems
Content-based retrieval
(4) Analysis of Learned Representations
We examine how the CNN organizes classes in feature space, comparing:
PCA of raw pixels
PCA of embeddings
t-SNE of embeddings
KMeans clustering results
This multi-angle analysis provides insights into what the network actually learns, and how representation quality affects downstream tasks.
