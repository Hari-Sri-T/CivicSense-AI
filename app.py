# --- Core Libraries ---
import os
import json
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import InferenceClient

# --- Document Processing ---
import pymupdf
import docx


# --- Google AI Integration (for answering only) ---
import google.generativeai as genai
from dotenv import load_dotenv

# --- Setup ---
load_dotenv()

# --- Initialize Client
HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
hf_client = InferenceClient(token=HF_TOKEN)

# Configure Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- GOOGLE CLOUD CONFIGURATION ---
API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)


# --- In-Memory Storage ---
document_chunk_store = {}
document_embeddings = []


# --- 1. DOCUMENT PROCESSING ---
def extract_pdf_with_structure(pdf_path: str):
    doc = pymupdf.open(pdf_path)
    results = []
    for pageno in range(doc.page_count):
        page = doc.load_page(pageno)
        text = page.get_text("text")
        if text.strip():
            results.append({"text": text, "page": pageno + 1})
    return results


def extract_docx_with_structure(docx_path: str):
    doc = docx.Document(docx_path)
    results = []
    for i, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        if text:
            results.append({"text": text, "para_idx": i})
    return results


def semantic_chunker(structured_parts: list[dict], max_tokens=350, overlap_tokens=50):
    """
    Simple semantic chunker: splits long documents into manageable chunks.
    """
    chunks = []
    buffer = ""
    for part in structured_parts:
        text = part["text"].strip()
        if not text:
            continue
        if len(buffer.split()) + len(text.split()) > max_tokens:
            if buffer:
                chunks.append({"text": buffer})
            buffer = text
        else:
            buffer += " " + text
    if buffer:
        chunks.append({"text": buffer})
    return chunks


# --- 2. AI & RAG CORE LOGIC ---
def embed_texts_hf(texts: list[str]) -> list[list[float]]:
    """
    Use Hugging Face Inference API for MiniLM embeddings.
    """
    embeddings = []
    for t in texts:
        try:
            resp = hf_client.feature_extraction(
                model="sentence-transformers/all-MiniLM-L6-v2",
                text=t,
                wait_for_model=True
            )
            # If response is token-level (list of lists), average
            if isinstance(resp, list) and len(resp) > 0 and isinstance(resp[0], list):
                import numpy as np
                vec = np.mean(resp, axis=0).tolist()
            else:
                vec = resp
            embeddings.append(vec)
        except Exception as e:
            print(f"HF embedding error: {e}")
            embeddings.append([0.0])  # avoid empty vector
    return embeddings



def store_in_memory(chunk_embeddings: list, chunks: list[dict]):
    global document_chunk_store, document_embeddings
    document_chunk_store = {}
    document_embeddings = []

    document_embeddings = np.array(chunk_embeddings)
    for i, chunk in enumerate(chunks):
        document_chunk_store[i] = chunk['text']

    print(f"Successfully stored {len(document_chunk_store)} chunks in memory.")
    return True


def retrieve_from_memory(query_embedding: list[float]) -> list[tuple[str, float]]:
    global document_chunk_store, document_embeddings
    if not document_chunk_store or len(document_embeddings) == 0:
        return []

    query_vec = np.array(query_embedding).reshape(1, -1)
    similarities = cosine_similarity(query_vec, document_embeddings)[0]
    top_indices = np.argsort(similarities)[-5:][::-1]

    results = []
    for i in top_indices:
        results.append((document_chunk_store[i], similarities[i]))

    return results


def generate_answer_prompt(query: str, top_chunks: list) -> str:
    sources_text = "\n\n".join([f"[Source {i+1}]:\n{txt}" for i, (txt, _) in enumerate(top_chunks)])
    prompt = f"""You are a helpful assistant. Use ONLY the provided sources to answer the user's QUERY. Respond in a valid JSON format with the keys: "answer" and "explanation".
- The "answer" key must contain a direct answer.
- The "explanation" key should briefly describe how you arrived at the answer.
If the sources are NOT relevant, the "answer" key must be "The provided documents do not contain specific information on this topic."

SOURCES:
---
{sources_text}
---
QUERY: {query}
"""
    return prompt


def llm_call_fn(prompt: str) -> dict:
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                response_mime_type="application/json"
            )
        )
    
        return json.loads(response.text)
    except Exception as e:
        print(f"Gemini API call failed: {e}")
        return {"answer": "Error: Could not generate a response.", "explanation": str(e)}


# --- 3. FLASK API ENDPOINTS ---
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        print("1. Extracting text...")
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext == ".pdf":
            structured_text = extract_pdf_with_structure(filepath)
        elif file_ext == ".docx":
            structured_text = extract_docx_with_structure(filepath)
        else:
            return jsonify({'error': 'Unsupported file type'}), 400

        print("2. Chunking document...")
        chunks = semantic_chunker(structured_text)
        chunk_texts = [c['text'].strip() for c in chunks if c['text'].strip()]

        if not chunk_texts:
            return jsonify({'error': 'No valid text found in document.'}), 400

        print("3. Embedding text chunks locally with MiniLM...")
        chunk_embeddings = embed_texts_hf(chunk_texts)
        if not chunk_embeddings:
            return jsonify({'error': 'Failed to embed chunks.'}), 500

        print(f"4. Storing {len(chunk_embeddings)} chunks in memory...")
        store_in_memory(chunk_embeddings, chunks)

        return jsonify({'message': 'File processed successfully.'}), 200
    return jsonify({'error': 'File processing failed'}), 500


@app.route('/query', methods=['POST'])
def ask_question():
    data = request.get_json()
    query = data.get('query')
    if not query:
        return jsonify({'error': 'No query provided'}), 400

    print("1. Embedding query locally with MiniLM...")
    query_embeddings = embed_texts_hf([query])
    if not query_embeddings:
        return jsonify({"error": "Failed to get embedding for the query."}), 500

    query_embedding = query_embeddings[0]

    print("2. Retrieving chunks from memory...")
    retrieved_chunks = retrieve_from_memory(query_embedding)

    print("3. Generating prompt...")
    prompt = generate_answer_prompt(query, retrieved_chunks)

    print("4. Calling Gemini LLM...")
    result = llm_call_fn(prompt)

    return jsonify(result), 200


if __name__ == '__main__':
    app.run(debug=True, port=5001)

# --- End of File ---

