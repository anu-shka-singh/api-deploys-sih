# from flask import Flask, render_template, request
# from summarizer import Summarizer

# app = Flask(__name__)

# @app.route("/", methods=["GET", "POST"])
# def index():
#     if request.method == "POST":
#         file_content = request.form["text"]
#         model = Summarizer()
#         res = model(file_content, min_length=60, max_length=150)
#         summary = ''.join(res)
#         return render_template("index.html", summary=summary)
#     return render_template("index.html", summary="")

# if __name__ == "__main__":
#     app.run(debug=True)
'''from flask import Flask, request, jsonify
from summarizer import Summarizer

app = Flask(__name__)
@app.route('/')
def index():
    return 'Welcome to the Text Summarizer API'


@app.route('/summarize/<data>', methods=['GET'])
def summarize_text(data):
    try:
        #data = request.get_json()
        list_of_words = data.split('%20')
        text_to_summarize = ' '.join(list_of_words)
        model = Summarizer()

        result = model(text_to_summarize, min_length=60, max_length=150)
        summary = ''.join(result)
        
        response = {
            'summary': summary
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)'''


from flask import Flask, request, jsonify
from summarizer import Summarizer
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Initialize the summarizer
model = Summarizer()

@app.route('/summarize', methods=['POST'])
def summarize():
    # Get the text input from the request
    file_content = request.form['text']
    # Generate the summary
    res = model(file_content, min_length=60, max_length=150)
    summary = ''.join(res)
    return jsonify({"summary": summary})

if __name__ == '__main__':
    app.run(debug=True)