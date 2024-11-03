import pandas as pd

# Load the data
orders_df = pd.read_csv('orders_data.csv')
products_df = pd.read_csv('products_data.csv')

# Create a pivot table (User-Product Matrix)
user_product_matrix = orders_df.pivot_table(index='customer_id', columns='product_id', values='purchase_count', fill_value=0)
print(user_product_matrix.head())

from sklearn.metrics.pairwise import cosine_similarity

# Calculate cosine similarity between products
product_similarity = cosine_similarity(user_product_matrix.T)
product_sim_df = pd.DataFrame(product_similarity, index=user_product_matrix.columns, columns=user_product_matrix.columns)
print(product_sim_df.head())
def get_recommendations(product_id, similarity_matrix, top_n=5):
    # Sort similar products by similarity score, excluding the product itself
    similar_scores = similarity_matrix[product_id]
    similar_products = similar_scores.sort_values(ascending=False).head(top_n + 1).index.tolist()
    return [prod for prod in similar_products if prod != product_id]

# Test the function
print("Recommendations for P001:", get_recommendations("P001", product_sim_df))

from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/', methods=['GET'])
def recommend():
    product_id = request.args.get('product_id')
    recommendations = get_recommendations(product_id, product_sim_df)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(port=5000)