from flask import Flask, request, jsonify, render_template
from transformers import pipeline
from flask_cors import CORS
import torch
import os
if os.path.exists("templates"):
    print("TEMPLATES DIR CONTENT:", os.listdir("templates"))
else:
    print("No templates directory found.")


app = Flask(__name__, template_folder="templates")
CORS(app)

@app.route("/")
def index():
    return render_template("index.html")

try:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    corrector = pipeline(
        "text2text-generation",
        model="t5-small",
        device=-1 if device == "cpu" else 0
    )
except Exception as e:
    print(f"Error loading model: {e}")
    corrector = None

@app.route("/check_grammar", methods=["POST"])
def check_grammar():
    if corrector is None:
        return jsonify({"error": "Model failed to load on the server."}), 500

    data = request.get_json()
    sentence = data.get("sentence", "").strip()
    if not sentence:
        return jsonify({"status": "MISSING_INPUT", "error": "No sentence provided."}), 400

    try:
        input_text = f"grammar: {sentence}"
        result = corrector(input_text, max_length=128, num_beams=4, early_stopping=True)
        corrected_sentence = result[0]["generated_text"].strip()

        if corrected_sentence.lower().startswith("grammar:"):
            corrected_sentence_for_check = corrected_sentence[8:].strip()
        else:
            corrected_sentence_for_check = corrected_sentence

        is_correct = (sentence.lower().strip(".!?, ") == corrected_sentence_for_check.lower().strip(".!?, "))

        if is_correct:
            status = "CORRECT"
        else:
            status = "INCORRECT"

        return jsonify({
            "status": status,
            "original": sentence,
            "corrected": corrected_sentence_for_check
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500