from flask import Flask, render_template, request, jsonify
from recommender import recommend_courses

app = Flask(__name__)

@app.route('/')
def index():

    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    data = request.get_json()      
    recommendations = recommend_courses(data)
    return jsonify(recommendations)  
if __name__ == '__main__':
    app.run(debug=True)
