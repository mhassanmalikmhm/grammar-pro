from flask import Flask, request, jsonify
from transformers import pipeline
from flask_cors import CORS
import torch

app = Flask(__name__)
CORS(app) # CORS enabled

# ===============================================
# Naya Model Loading (T5 for correction)
# ===============================================
try:
    # Device setup
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device set to use {device}")
    
    print("Loading T5-small model for Grammar Correction...")
    
    # Correction pipeline
    corrector = pipeline(
        "text2text-generation", 
        model="t5-small",
        device=-1 if device == "cpu" else 0
    )
    print("T5-small model loaded successfully.")

except Exception as e:
    print(f"Error loading model: {e}")
    corrector = None

# -----------------------------------------------

@app.route('/check_grammar', methods=['POST'])
def check_grammar():
    if corrector is None:
        return jsonify({"error": "Model failed to load on the server."}), 500

    data = request.get_json()
    sentence = data.get('sentence', '').strip()
    
    if not sentence:
        return jsonify({"status": "MISSING_INPUT", "error": "No sentence provided."}), 400

    # Prediction Logic
    try:
        # T5 model ko instruction dena
        input_text = f"grammar: {sentence}"
        
        # Correction generation
        result = corrector(
            input_text, 
            max_length=128, 
            num_beams=4, 
            early_stopping=True
        )
        
        corrected_sentence = result[0]['generated_text'].strip()

        # Logic to remove 'grammar: ' prefix for accurate comparison
        # Yeh woh fix hai jo humne pichle conversation mein discuss kiya tha
        display_corrected = corrected_sentence
        if corrected_sentence.lower().startswith("grammar:"):
            corrected_sentence_for_check = corrected_sentence[8:].strip()
        else:
            corrected_sentence_for_check = corrected_sentence
        
        # Check: Agar original aur corrected sentences same hain (case-insensitive) to CORRECT
        is_correct = (sentence.lower().strip('.!?, ') == corrected_sentence_for_check.lower().strip('.!?, '))
        
        status = "CORRECT" if is_correct else "PROBLEM"

        return jsonify({
            "status": status,
            "original": sentence,
            "corrected": display_corrected, # Frontend ko pura corrected text bhej rahe hain
            "explanation": "T5-small Model correction applied."
        })

    except Exception as e:
        # Agar prediction mein koi masla aaye, to yeh block chalega
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)