from flask import Flask, request, jsonify
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Model
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

# Register the custom loss function
@tf.keras.utils.register_keras_serializable()
def mse(y_true, y_pred):
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

# Load and preprocess data
data = pd.read_csv('product.csv', delimiter=';')
data = data[['id', 'name', 'order_click', 'history_view_product', 'min_order']]

# Normalize data
scaler = StandardScaler()
data[['order_click', 'history_view_product', 'min_order']] = scaler.fit_transform(data[['order_click', 'history_view_product', 'min_order']])

# Convert data to numpy array
product_features = data[['order_click', 'history_view_product', 'min_order']].values

# Load pre-trained Keras model
model = load_model('model1.h5', custom_objects={'mse': mse})

# Define the encoder model
encoder_model = Model(inputs=model.input, outputs=model.layers[-2].output)

# Compile the model with metrics
model.compile(optimizer='adam', loss=mse, metrics=['accuracy'])

# Get latent representation of products
product_embeddings = encoder_model.predict(product_features)

# Load SentenceTransformer model
model_fitur_search = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def get_recommendations(product_id, embeddings, product_data, top_k=5):
    # Calculate cosine similarity with all products
    similarities = cosine_similarity([embeddings[product_id]], embeddings)[0]
    # Get indices of top_k most similar products
    similar_indices = similarities.argsort()[-top_k-1:-1][::-1]
    # Get names of recommended products
    recommended_product_names = product_data.iloc[similar_indices]['name'].tolist()
    return recommended_product_names

def recommend_similar_products(clicked_product_name, product_data, top_k=5):
    # Convert clicked product name to lower case
    clicked_product_name = clicked_product_name.lower()

    # Get embedding for the clicked product name
    clicked_product_embedding = model_fitur_search.encode([clicked_product_name], convert_to_tensor=True)

    # Convert all product names in data to lower case
    product_names = [name.lower() for name in product_data['name'].tolist()]

    # Get embeddings for all product names
    product_embeddings = model_fitur_search.encode(product_names, convert_to_tensor=True)

    # Compute similarity between clicked product name and all other product names
    similarity_scores = util.pytorch_cos_sim(clicked_product_embedding, product_embeddings)[0]

    # Sort products by similarity score
    similar_products_indices = similarity_scores.argsort(descending=True)

    # Get product names similar by embedding similarity
    similar_products_by_embedding = [product_names[i] for i in similar_products_indices[:top_k]]

    # Find product names containing the searched substring
    similar_products_by_substring = [name for name in product_names if clicked_product_name in name]
    
    # Combine results from both approaches and ensure uniqueness
    seen = set()
    combined_results = []
    for name in similar_products_by_substring + similar_products_by_embedding:
        if name not in seen:
            seen.add(name)
            combined_results.append(name)
            if len(combined_results) >= top_k:
                break

    return combined_results

@app.route('/')
def home():
    return "Welcome to the Product Recommendation API!"

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/recommend/content', methods=['POST'])
def recommend_content():
    try:
        arg = request.get_json()
        product_id = arg['item_id']
        recommendations = get_recommendations(product_id, product_embeddings, data)
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/recommend/search', methods=['POST'])
def recommend_search():
    try:
        arg = request.get_json()
        clicked_product_name = arg.get('product_name', '')
        top_k = arg.get('top_k', 5)
        recommendations = recommend_similar_products(clicked_product_name, data, top_k)
        return jsonify(recommendations)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
