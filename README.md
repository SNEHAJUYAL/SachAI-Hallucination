# SachAI — AI Hallucination Detector

SachAI is a forensic tool that detects hallucinations in AI-generated responses by comparing them against a ground-truth source document. It breaks the response into atomic claims, runs NLI-based cross-examination against the source, and produces a per-claim verdict with a composite risk score.

---

## How It Works

1. **Claim Extraction** — The AI response is split into atomic claims using spaCy's dependency parser (coordinating conjunctions are used to sub-split sentences only when both sides contain a verb).
2. **Semantic Alignment** — Each claim is embedded with `all-MiniLM-L6-v2` and matched to the most similar source sentence via cosine similarity.
3. **NLI Cross-Examination** — The matched source sentence and claim are passed to `cross-encoder/nli-deberta-v3-small` to determine entailment, contradiction, or neutrality.
4. **Taxonomy Classification** — Each claim is labelled:
   - `Verified Fact` — entailed by source
   - `Contradiction` — contradicted with high confidence
   - `Numeric Drift` — claim contains a number not found in the source
   - `Safe / Weak Inference` — neutral NLI but semantically close
   - `Likely Hallucination` — neutral NLI and low similarity
   - `Grounded / Ungrounded Opinion` — for opinion-type claims
   - `Relevant / Hallucinated Suggestion` — for suggestion-type claims
5. **Risk Scoring** — A risk index (0–100%) is computed from the fraction of unsupported claims. Responses above 60% risk are flagged as `HALLUCINATED`.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React + TypeScript, Vite, Tailwind CSS, shadcn/ui |
| Backend | Python, Flask |
| NLI Model | `cross-encoder/nli-deberta-v3-small` (HuggingFace) |
| Embeddings | `all-MiniLM-L6-v2` (SentenceTransformers) |
| NLP | spaCy `en_core_web_sm` |
| Deployment | Vercel (frontend + backend) |

---

## Project Structure

```
sachai/
├── backend/
│   ├── app.py              # Original Flask API (port 5000)
│   ├── hallucinator.py     # Production service with /health & /intercept endpoints (port 5001)
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── api/analyze.ts          # API client + response mapping
│       ├── components/dashboard/   # UI components (gauge, heatmap, claims list, etc.)
│       ├── types/dashboard.ts      # TypeScript types
│       └── App.tsx                 # Root component
└── vercel.json
```

---

## Getting Started

### Backend

```bash
cd backend
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python hallucinator.py
```

The service starts on port `5001` by default. Override with:

```bash
HALLUCINATOR_PORT=5001 HALLUCINATION_THRESHOLD=0.6 python hallucinator.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Set the backend URL via environment variable (defaults to `http://localhost:5000`):

```bash
VITE_API_URL=http://localhost:5001 npm run dev
```

---

## API Reference

### `POST /analyze`

Full claim-level analysis.

**Request**
```json
{ "source": "ground truth text", "response": "AI generated text" }
```

**Response**
```json
{
  "source_sentences": ["..."],
  "ai_claims": ["..."],
  "results": [
    {
      "claim": "...",
      "intent": "FACT",
      "taxonomy": "Verified Fact",
      "similarity": 0.87,
      "alibi": "matched source sentence",
      "nli": "Entailed",
      "confidence": 0.94
    }
  ],
  "verdict": "FAITHFUL",
  "faithfulness_score": 0.85,
  "flagged_claims": []
}
```

### `POST /intercept`

Fast-path verdict only — used for middleware integration.

### `GET /health`

Liveness check. Returns `{ "status": "ok", "device": "cpu" | "cuda" }`.

---

## Dashboard

The frontend visualises results across five panels:

- **Veracity Score Gauge** — overall trust score
- **Hallucination Taxonomy** — breakdown of claim categories
- **Evidence Viewer** — source segments with matched claims highlighted
- **Alignment Heatmap** — cosine similarity matrix (claims × source sentences)
- **Claims List** — per-claim verdict, NLI scores, and matched evidence
