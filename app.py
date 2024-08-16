from flask import Flask, request, render_template, jsonify
from finetunerun import translate_text
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
# Load your fine-tuned model (assuming PyTorch)
# Replace this with the actual loading code for your model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/interact', methods=['POST'])
def interact():
    data = request.json
    source = data.get('source')
    input_text = data.get('input_text')
    print(source)
    print(input_text)    
    # Example of how you might process the input
    # Replace this with your model's interaction logic
    response = model_process(input_text, source)
    
    return jsonify({"response": response})

def model_process(input_text, source):
    # Implement the logic to process the input with your model
    # Example: concatenate source and input_text and use as input to the model
    # Replace with your actual model inference logic
    combined_input = f"{source}: {input_text}"
    # Example output, replace this with your model's actual output
    output = translate_text(input_text, source)
    return output

if __name__ == '__main__':
    app.run(debug=True)
