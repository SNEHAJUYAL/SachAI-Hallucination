import os
import spacy
import time
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:5173")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_methods=["POST"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    source: str
    response: str

# --- Model Init ---
nlp_ner = spacy.load("en_core_web_sm")

MODEL_PATH = "/kaggle/input/notebooks/anirbandasbit/sentenceclassifier/output/model-best"
nlp_intent = spacy.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else nlp_ner

EMBED_URL = "https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2/pipeline/feature-extraction"
NLI_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli/pipeline/zero-shot-classification"

HF_TOKEN = os.environ.get("HF_TOKEN")
HF_HEADERS = {"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"}

# --- Pipeline Functions ---
def has_verb(tokens):
    return any(tok.pos_ in {"VERB", "AUX"} for tok in tokens)

# calculate cosine similarity
def cos_sim_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix between rows of a and rows of b."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a_norm @ b_norm.T  # shape: (len_a, len_b)

def extract_claims(text):
    split_words = {"and", "but", "also", "however"}
    doc = nlp_ner(text)
    claims = []
    for sent in doc.sents:
        current_chunk = []
        tokens = list(sent)
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.text.lower() in split_words and token.dep_ == "cc":
                if has_verb(current_chunk) and has_verb(tokens[i + 1 :]):
                    if current_chunk:
                        claims.append(" ".join(t.text for t in current_chunk).strip())
                        current_chunk = []
                    i += 1
                    continue
            current_chunk.append(token)
            i += 1
        if current_chunk:
            claims.append(" ".join(t.text for t in current_chunk).strip())
    return [c for c in claims if len(c) > 5]

def classify_intent(text):
    if not os.path.exists(MODEL_PATH):
        return "FACT"
    doc = nlp_intent(text)
    return max(doc.cats, key=doc.cats.get)

def get_embeddings(sentences: list[str]) -> np.ndarray:
    payload = {
        "inputs": sentences,
        "options": {"wait_for_model": True}
    }
    for attempt in range(3):
        resp = requests.post(EMBED_URL, headers=HF_HEADERS, json=payload)
        if resp.status_code != 200 or not resp.text.strip():
            print(f"[HF Embeddings] Attempt {attempt+1} failed: HTTP {resp.status_code} | {resp.text!r}")
            time.sleep(2 ** attempt)
            continue
        try:
            data = resp.json()
            return np.array(data, dtype=np.float32)
        except Exception as e:
            print(f"[HF Embeddings] Parse error: {e} | {resp.text!r}")
            time.sleep(2 ** attempt)
    raise RuntimeError("HF Embeddings API failed after 3 attempts.")

def get_nli_verdict(source_sentence: str, ai_claim: str):
    # bart-large-mnli uses zero-shot-classification pipeline
    # We classify the CLAIM against the SOURCE as context
    payload = {
        "inputs": ai_claim,
        "parameters": {
            "candidate_labels": ["entailment", "neutral", "contradiction"],
            "hypothesis_template": f"Based on '{source_sentence}', this statement is {{}}."
        },
        "options": {"wait_for_model": True}
    }

    for attempt in range(3):
        resp = requests.post(NLI_URL, headers=HF_HEADERS, json=payload)

        if resp.status_code != 200 or not resp.text.strip():
            print(f"[HF NLI] Attempt {attempt+1} failed: HTTP {resp.status_code} | {resp.text!r}")
            time.sleep(2 ** attempt)
            continue

        try:
            data = resp.json()
            # Response: [{'label': 'contradiction', 'score': 0.5283074378967285}, {'label': 'neutral', 'score': 0.31749388575553894}, {'label': 'entailment', 'score': 0.15419872105121613}]
            label_map = {
                "contradiction": "Contradicted",
                "entailment": "Entailed",
                "neutral": "Neutral"
            }

            top = data[0]
            return label_map.get(top["label"].lower(), top["label"]), round(top["score"], 4)

        except Exception as e:
            print(f"[HF NLI] Parse error: {e} | {resp.text!r}")
            time.sleep(2 ** attempt)

    raise RuntimeError("HF NLI API failed after 3 attempts.")

def check_numeric_drift(ai_claim_text, top_source_sentence):
    NUMERIC_LABELS = {"MONEY", "PERCENT", "DATE", "TIME", "CARDINAL", "QUANTITY"}
    claim_numbers = [e.text for e in nlp_ner(ai_claim_text).ents if e.label_ in NUMERIC_LABELS]
    if not claim_numbers:
        return "PASS"
    for num in claim_numbers:
        if num not in top_source_sentence:
            return f"Drift: '{num}' not found."
    return "PASS"

def build_alignment_matrix(ai_claims, source_sentences):
    # claim_emb = embedder.encode(ai_claims, convert_to_tensor=True, device=device)
    # source_emb = embedder.encode(source_sentences, convert_to_tensor=True, device=device)
    claim_emb = get_embeddings(ai_claims)
    source_emb = get_embeddings(source_sentences)
    matrix = cos_sim_numpy(claim_emb, source_emb)
    result = []
    for i in range(len(ai_claims)):
        max_idx = int(np.argmax(matrix[i]))
        max_score = float(matrix[i][max_idx])
        result.append({
            "S_Max": round(max_score, 4),
            "Source_Index": max_idx,
            "Matched_Sentence": source_sentences[max_idx],
            "matrix_row": [round(float(v), 4) for v in matrix[i]],
        })
    return result

def evaluate_response(ai_claims, source_sentences):
    matrix_data = build_alignment_matrix(ai_claims, source_sentences)
    results = []

    for i, claim in enumerate(ai_claims):
        intent = classify_intent(claim)
        alibi = matrix_data[i]["Matched_Sentence"]
        s_max = matrix_data[i]["S_Max"]
        matrix_row = matrix_data[i]["matrix_row"]
        source_idx = matrix_data[i]["Source_Index"]

        ev = {
            "claim": claim,
            "intent": intent,
            "taxonomy": "TBD",
            "similarity": s_max,
            "alibi": alibi,
            "source_index": source_idx,
            "matrix_row": matrix_row,
        }

        # --- FACT HANDLING ---
        if intent.upper() == "FACT":
            nli, conf = get_nli_verdict(alibi, claim)

            # --- Forgiving Logic ---
            if nli == "Entailed":
                taxonomy = "Verified Fact"

            elif nli == "Contradicted":
                # Only penalize strong contradictions
                if conf > 0.75:
                    taxonomy = "Contradiction"
                else:
                    taxonomy = "Possibly Misaligned"

            else:  # Neutral
                if s_max > 0.75:
                    taxonomy = "Safe Inference"
                elif s_max > 0.5:
                    taxonomy = "Weak Inference"
                else:
                    taxonomy = "Likely Hallucination"

            ev.update({
                "taxonomy": taxonomy,
                "nli": nli,
                "confidence": conf
            })

        # --- OPINION HANDLING ---
        elif intent.upper() == "OPINION":
            if s_max > 0.75:
                ev["taxonomy"] = "Grounded Opinion"
            elif s_max > 0.5:
                ev["taxonomy"] = "Loosely Grounded Opinion"
            else:
                ev["taxonomy"] = "Ungrounded Opinion"

        # --- SUGGESTION / OTHER ---
        else:
            if s_max > 0.7:
                ev["taxonomy"] = "Relevant Suggestion"
            elif s_max > 0.4:
                ev["taxonomy"] = "Weak Suggestion"
            else:
                ev["taxonomy"] = "Irrelevant / Hallucinated Suggestion"

        results.append(ev)

    return results

@app.post("/analyze")
async def analyze(body: AnalyzeRequest, request: Request):
    # Check request origin for safety
    origin = request.headers.get("origin")
    if origin != FRONTEND_URL:
        raise HTTPException(status_code=403, detail="Forbidden origin")

    source = body.source.strip()
    response = body.response.strip()
    if not source or not response:
        raise HTTPException(status_code=400, detail="Both 'source' and 'response' are required.")

    source_sentences = extract_claims(source)
    ai_claims = extract_claims(response)

    if not source_sentences or not ai_claims:
        raise HTTPException(status_code=422, detail="Could not extract claims from the provided text.")

    results = evaluate_response(ai_claims, source_sentences)
    return {
        "source_sentences": source_sentences,
        "ai_claims": ai_claims,
        "results": results,
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)