import type { DashboardData, Claim, ClaimType, SourceSegment } from '@/types/dashboard';

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';
const FRONTEND_SECRET = import.meta.env.VITE_FRONTEND_SECRET;

interface BackendResult {
  claim: string;
  intent: string;
  taxonomy: string;
  similarity: number;
  alibi: string;
  source_index: number;
  matrix_row: number[];
  nli?: string;
  confidence?: number;
  error?: string;
}

interface BackendResponse {
  source_sentences: string[];
  ai_claims: string[];
  results: BackendResult[];
}

function toClaimType(taxonomy: string): ClaimType {
  if (taxonomy === 'Verified Fact') return 'FAITHFUL';
  if (taxonomy === 'Contradiction' || taxonomy === 'Numeric Drift') return 'CONTRADICTED';
  if (taxonomy === 'Extrapolation' || taxonomy.includes('Grounded')) return 'EXTRAPOLATED';
  return 'FABRICATED';
}

function toNliScores(result: BackendResult) {
  const nli = result.nli?.toLowerCase();
  const conf = result.confidence ?? result.similarity;
  if (nli === 'entailed') return { entailment: conf, contradiction: (1 - conf) / 2, neutral: (1 - conf) / 2 };
  if (nli === 'contradicted') return { entailment: (1 - conf) / 2, contradiction: conf, neutral: (1 - conf) / 2 };
  return { entailment: (1 - conf) / 2, contradiction: (1 - conf) / 2, neutral: conf };
}

export async function analyzeText(source: string, response: string): Promise<DashboardData> {
  const res = await fetch(`${API_URL}/analyze`, {
    method: 'POST',
    headers: { 
      'Content-Type': 'application/json',
      'x-api-key': FRONTEND_SECRET
    },
    body: JSON.stringify({ source, response }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error((err as { error?: string }).error ?? `Request failed: ${res.status}`);
  }

  const data: BackendResponse = await res.json();
  const { source_sentences, ai_claims, results } = data;
  console.log(ai_claims)

  const claims: Claim[] = results.map((r, i) => ({
    id: `c${i + 1}`,
    text: r.claim,
    type: toClaimType(r.taxonomy),
    confidence: r.confidence ?? r.similarity,
    sourceRef: r.alibi ? r.alibi.slice(0, 40) + '…' : undefined,
    nliScores: toNliScores(r),
  }));

  const sourceSegments: SourceSegment[] = source_sentences.map((text, si) => ({
    id: `s${si + 1}`,
    text,
    matchedClaims: results
      .map((r, ci) => ({ r, ci }))
      .filter(({ r }) => r.source_index === si)
      .map(({ ci }) => `c${ci + 1}`),
  }));

  const alignmentMatrix: number[][] = results.map((r) => r.matrix_row);

  const taxonomy = {
    faithful: claims.filter((c) => c.type === 'FAITHFUL').length,
    extrapolated: claims.filter((c) => c.type === 'EXTRAPOLATED').length,
    fabricated: claims.filter((c) => c.type === 'FABRICATED').length,
    contradicted: claims.filter((c) => c.type === 'CONTRADICTED').length,
  };

  const unsupported = taxonomy.fabricated + taxonomy.contradicted;
  const riskIndex = Math.round((unsupported / Math.max(claims.length, 1)) * 100);
  const trustScore = 100 - riskIndex;
  const verdict = riskIndex >= 60 ? 'HALLUCINATED' : riskIndex >= 30 ? 'PARTIAL' : 'FAITHFUL';
  const riskLevel = riskIndex >= 80 ? 'CRITICAL' : riskIndex >= 60 ? 'HIGH' : riskIndex >= 40 ? 'MEDIUM' : 'LOW';

  return {
    sourceText: source,
    aiResponse: response,
    analysis: {
      verdict,
      trustScore,
      riskIndex,
      riskLevel,
      claims,
      sourceSegments,
      alignmentMatrix,
      taxonomy,
      processingTime: 0,
      modelUsed: 'cross-encoder/nli-deberta-v3-small',
    },
  };
}
