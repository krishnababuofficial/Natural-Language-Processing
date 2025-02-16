
# Rotten Tomatoes Movie Sentiment Analysis with FastText: Supervised and Unsupervised Methods

This project explores sentiment analysis on the Rotten Tomatoes Movies dataset using FastText, a library for word embeddings and text classification. The project implements both supervised and unsupervised learning methods to analyze movie sentiment.  The focus is on comparing the performance of these two approaches.


## Project Overview
This project demonstrates:
1. **Data Loading and Preprocessing:** Loading and cleaning the Rotten Tomatoes Movies dataset, preparing it for analysis.
2. **Supervised FastText Training:** Training a supervised FastText model to classify movie sentiment (positive/negative) using movie descriptions and critics' consensus.
3. **Unsupervised FastText Training:** Training an unsupervised FastText model to learn word embeddings from the text data, capturing semantic relationships between words.
4. **Sentiment Analysis Comparison:**  Comparing the performance of the supervised FastText model for sentiment prediction against qualitative insights gained from the unsupervised model’s word embeddings.
5. **Performance Evaluation:**  Assessing and visualizing the performance of the supervised model using relevant metrics.

## Data
The project utilizes the `Rotten_Tomatoes_Movies3` dataset [Link to dataset on GitHub if applicable, or description of location within repository]. This dataset contains the following relevant columns:

* `movie_title`: Title of the movie.
* `movie_info`: General information about the movie.
* `critics_consensus`: Summary of critics' reviews.
* `tomatometer_status`: Categorical variable indicating the Tomatometer rating ("Rotten," "Fresh," or "Certified Fresh").
* `audience_rating`: Audience rating for the movie.

The data undergoes preprocessing using spaCy, including:

* Text Cleaning: Removing punctuation and special characters.
* Tokenization: Splitting text into individual words.
* Lemmatization: Reducing words to their dictionary form.
* Stop Word Removal: Eliminating frequent, less informative words.

The preprocessed data is split into training, validation, and test sets [Explain how the data is split, e.g., 70/15/15 or 80/10/10].

## Methodology
**1. Supervised FastText:**
* A supervised FastText model is trained on the preprocessed `movie_info` and `critics_consensus` text, using a derived sentiment label (positive/negative) from `tomatometer_status`.
* Model hyperparameters are tuned using FastText's automatic optimization.  The specific `autotuneMetric` used is [mention the specific metric used, e.g., `precisionAtRecall:30`].
* Model performance is evaluated on the validation and test sets using [list metrics: e.g., Precision@Recall:30, Accuracy, Precision, Recall, F1-score].

**2. Unsupervised FastText:**
* An unsupervised FastText model is trained on the preprocessed text data to learn word embeddings.
* Nearest neighbor analysis is then performed to identify words semantically similar to terms frequently associated with positive and negative reviews.  This analysis helps to understand contextual relationships and reveals how sentiment is linguistically expressed in the data.

## Results
**Quantitative Results (Supervised Model):**
* **Validation Set:**  0.7048054919908466
* **Test Set:** 0.697986577181208

**Qualitative Results (Unsupervised Model):**
* The model_unsupervised suggests that words like "revenge," "survivor," "slavery," "terrain," and others are semantically close to "alien" based on the word vectors learned from the unsupervised training data.

## Technologies Used
* Python
* pandas, NumPy, spaCy, FastText, scikit-learn (if used for additional classifiers)
* Jupyter Notebook, Git

### License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details