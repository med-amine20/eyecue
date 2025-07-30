from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline, set_seed
import os

# Initialiser Spark NLP (avec sécurité)
try:
    import sparknlp
    from pyspark.sql import SparkSession
    from sparknlp.base import *
    from sparknlp.annotator import *

    if not os.environ.get("JAVA_HOME"):
        os.environ["JAVA_HOME"] = r"C:\JAVA\jdk1.8.0_202"

    spark = sparknlp.start()
    spark_nlp_ready = True
except Exception as e:
    print("[ERREUR SPARK NLP] :", e)
    spark_nlp_ready = False
    spark = None

# Flask app
app = Flask(__name__)
CORS(app)

# Générateur GPT-2 français
generator = pipeline('text-generation', model='dbddv01/gpt2-french-small')
set_seed(42)

# Tokenisation NLP via Spark NLP
def tokenize_with_sparknlp(text):
    if not spark_nlp_ready:
        return []

    data = spark.createDataFrame([[text]]).toDF("text")
    document_assembler = DocumentAssembler().setInputCol("text").setOutputCol("document")
    tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("token")

    pipeline = Pipeline(stages=[document_assembler, tokenizer])
    model = pipeline.fit(data)
    result = model.transform(data)

    tokens = result.selectExpr("explode(token.result) as token").rdd.map(lambda r: r[0]).collect()
    return tokens

# Route /suggest
@app.route('/suggest', methods=['POST'])
def suggest():
    data = request.json
    text = data.get('text', '').strip()

    if not text:
        return jsonify({'suggestions': [], 'tokens': []})

    try:
        outputs = generator(
            text,
            max_new_tokens=6,         # court, précis
            num_return_sequences=5,   # diversité
            do_sample=True,
            top_k=50,
            top_p=0.92,
            pad_token_id=50256
        )

        suggestions = set()
        for out in outputs:
            generated = out['generated_text'][len(text):].strip()

            # Nettoyage de fin de phrase
            for sep in ['.', '!', '?', '\n']:
                if sep in generated:
                    generated = generated.split(sep)[0]

            # Extraction du 1er mot ou groupe de 2 mots
            words = generated.split()
            if words:
                short_suggestion = ' '.join(words[:2])  # 1 ou 2 mots
                if short_suggestion.lower() not in suggestions:
                    suggestions.add(short_suggestion.lower())

        tokens = tokenize_with_sparknlp(text)

        return jsonify({
            'suggestions': list(suggestions),
            'tokens': tokens
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Lancer serveur Flask
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
