# IMDB Sentiment Analysis with FastText: Supervised and Unsupervised Methods

This repository explores sentiment analysis on the IMDB dataset using FastText, a powerful library for word embeddings and text classification. The code implements both supervised and unsupervised methods to analyze sentiment in movie reviews.

### Project Overview

**The project demonstrates :**
* **Data Loading and Preprocessing :** Loading and cleaning the IMDB dataset, preparing it for analysis.
* **Supervised FastText Training :** Training a supervised FastText model for sentiment classification (positive/negative).
* **Unsupervised FastText Training :** Training an unsupervised FastText model to learn word embeddings, capturing semantic relationships between words.
* **Traditional Machine Learning Models :** Using word embeddings from the unsupervised FastText model as features for training traditional classifiers (Multinomial Naive Bayes, K-Nearest Neighbors, Random Forest).
* **Performance Evaluation :** Assessing the accuracy, precision, recall, and confusion matrix visualizations of each model.

### Key Features

* **FastText Implementation :** Leverages both supervised and unsupervised FastText methods for sentiment analysis.
* **Comparative Study :** Compares the performance of FastText with traditional machine learning models.
* **Visualization :** Provides clear visualizations of model performance using confusion matrices.

### Getting Started

1. **Clone the repository :**
   ```bash
    git clone https://github.com/your-username/imdb-sentiment-analysis-fasttext.git
   ```
Use code with caution.

2.  **Install dependencies :**
    ```bash
    export pip install pandas numpy seaborn matplotlib fasttext scikit-learn spacy
    ```
Use code with caution.

3. **Download the IMDB dataset :**
    ```bash
    export Download the "IMDB Dataset.csv" file (available on Kaggle or similar datasets) and place it in the same directory as the code.
    ```
4. **Run the code :**
    ```bash
    export python sentiment_analysis.py
    ```
Use code with caution.
Bash

### Output
The code will output:
* **Performance Metrics :** Accuracy, precision, and recall for each model (FastText, Naive Bayes, KNN, Random Forest).
* **Predictions :** Sentiment predictions for sample review text using the trained FastText models.
* **Confusion Matrices :** Visualizations of model performance showing correct and incorrect classifications.

## Contributions
Contributions to this project are welcome! If you find issues or have improvements, feel free to fork the repository and submit pull requests.

### License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details