from flask import Flask, request, jsonify, render_template
import spacy
from transformers import BartForConditionalGeneration, BartTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim import corpora, models
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask
app = Flask(__name__)

# Load required models
nlp = spacy.load("en_core_web_sm")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

# Function to remove stop words using spaCy
def remove_stop_words(text):
    doc = nlp(text)
    return " ".join([token.text for token in doc if not token.is_stop])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize_text():
    content = request.json
    text = content['text']
    inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = model.generate(inputs.input_ids, num_beams=4, min_length=30, max_length=150, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return jsonify({"summary": summary})

@app.route('/extract', methods=['POST'])
def extract_features_entities():
    content = request.json
    text = content['text']
    cleaned_text = remove_stop_words(text)  # Remove stop words
    doc = nlp(text)
    entities = {ent.text: ent.label_ for ent in doc.ents}
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([cleaned_text])  # Using cleaned text
    features = vectorizer.get_feature_names_out()
    return jsonify({"entities": entities, "features": list(features)})

@app.route('/lda', methods=['POST'])
def lda_clustering():
    content = request.json
    text = content['text']

    # Summarize the text
    inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
    summary_ids = model.generate(inputs.input_ids, num_beams=4, min_length=30, max_length=150, early_stopping=True)
    summarized_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Apply LDA on the summarized text
    cleaned_text = remove_stop_words(summarized_text)  # Remove stop words from summarized text
    tokenized_list = [word_tokenize(doc) for doc in sent_tokenize(cleaned_text)]
    id2word = corpora.Dictionary(tokenized_list)
    corpus = [id2word.doc2bow(text) for text in tokenized_list]
    lda_model = models.LdaModel(corpus, id2word=id2word, num_topics=2)
    lda_topics = lda_model.print_topics()
    return jsonify({"lda_topics": lda_topics})

@app.route('/semantic_similarity', methods=['POST'])
def semantic_clustering():
    content = request.json
    text = content['text']
    sentences = sent_tokenize(text)
    cleaned_sentences = [remove_stop_words(sent) for sent in sentences]  # Remove stop words
    doc_vectors = [nlp(sent).vector for sent in cleaned_sentences]  # Using cleaned sentences
    similarity_matrix = cosine_similarity(doc_vectors)
    groups = []
    for i, sentence in enumerate(cleaned_sentences):  # Using cleaned sentences
        group = [cleaned_sentences[j] for j in range(len(cleaned_sentences)) if similarity_matrix[i][j] > 0.8]
        groups.append(group)
    return jsonify({"groups": groups})

if __name__ == '__main__':
    app.run(debug=True)
