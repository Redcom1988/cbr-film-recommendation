from flask import Flask, request, jsonify, render_template
from recommend import CBRRecommender

app = Flask(__name__)
recommender = CBRRecommender()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    data = request.json
    query = data.get('query', '')
    user_id = data.get('user_id', None)
    
    if user_id:
        try:
            user_id = int(user_id)
        except ValueError:
            user_id = None
            
    if not query:
        return jsonify({'error': 'Query cannot be empty'}), 400
        
    results = recommender.recommend(query_text=query, user_id=user_id, top_k=6)
    return jsonify({'results': results})

@app.route('/api/feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    query = data.get('query', '')
    accepted_ids = data.get('accepted_ids', [])
    rejected_ids = data.get('rejected_ids', [])
    
    recommender.revise_and_retain(query, accepted_ids, rejected_ids)
    return jsonify({'status': 'success', 'message': 'Feedback received and case retained!'})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
