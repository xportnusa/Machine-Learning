# Machine-Learning
## Summary
To enhance user engagement on our platform, XportNusa, we've integrated a machine learning-powered recommender system. This system analyzes user behavior such as click frequency, purchase history, and viewed product counts. By leveraging this data, XportNusa can provide tailored recommendations to buyers, suggesting relevant products that align with their interests and purchasing patterns. This approach not only personalizes the user experience but also facilitates informed decision-making, ultimately optimizing the import and export process for our users. With continuous feedback and refinement based on user interactions, our recommender system aims to deliver valuable insights and enhance user satisfaction across the platform.

## ML Model Details 


## model1_content based filterting.ipynb

### A. Data Exploratory 
1. Load and Read Data
2. Explore and Analyze Data

### B. Data Preparation
1. Column Selection
2. Normalization
3. Convert to Numpy Array

### C. Modelling Process 
1. Autoencoder Setup
2. Model Training
3. Generate Product Embeddings
4. Product Recommendations

### D. Evaluation 
1. Save Model


## model2_fitur search.ipynb
### A. Install Library and Import Modules
Installs necessary libraries and imports required modules, including SentenceTransformer for a pre-trained model and util for utility functions.
### B. Load Pre-trained Model
Loads the pre-trained model paraphrase-xlm-r-multilingual-v1 from Sentence Transformers. This model is used to generate embeddings that represent the meaning of sentences or phrases.
### C. recommend_similar_products Function
1. This function recommends similar products based on a clicked product name.
2. Converts the clicked product name to lowercase for consistency.
3. Generates embeddings for the clicked product name and all product names in the dataset.
4. Calculates cosine similarity between the clicked product embedding and all other product embeddings.
5. Sorts products based on similarity score and combines results from two approaches: embedding similarity and substring matching of the clicked product name.
6. Returns a list of recommended product names similar to the clicked product name, limited by the top_k parameter.

## app.py
