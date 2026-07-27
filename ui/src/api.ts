/** Typed client for the voqx registry API. */

export interface IdentitySummary {
  voice_id: string;
  display_name: string;
  aliases: string[];
  recognition: boolean;
  synthesis: boolean;
  wake: boolean;
  recognition_samples: number;
  synthesis_references: number;
}

export interface RegistryResponse {
  root: string;
  identities: IdentitySummary[];
}

export interface SampleAnalysis {
  index: number;
  audio: string | null;
  own_score: number;
  leave_one_out: number | null;
  best_voice_id: string;
  best_display_name: string;
  best_score: number;
  misattributed: boolean;
  weak: boolean;
}

export interface IdentityAnalysis {
  voice_id: string;
  display_name: string;
  engine: string;
  mean_leave_one_out: number | null;
  samples: SampleAnalysis[];
}

export interface SimilarityRow {
  voice_id: string;
  display_name: string;
  scores: number[];
}

export interface AnalysisResponse {
  engine: string;
  analyses: IdentityAnalysis[];
  matrix: { voice_ids: string[]; rows: SimilarityRow[] };
}

async function json<T>(input: string, init?: RequestInit): Promise<T> {
  const response = await fetch(input, init);
  if (!response.ok) {
    const body = (await response.json().catch(() => null)) as { error?: string } | null;
    throw new Error(body?.error ?? `${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

export const api = {
  registry: () => json<RegistryResponse>("/api/registry"),
  analysis: () => json<AnalysisResponse>("/api/analysis"),
  audioUrl: (path: string) => `/api/audio?path=${encodeURIComponent(path)}`,
  dropSample: (voiceId: string, index: number, deleteAudio: boolean) =>
    json<{ ok: boolean }>(
      `/api/identities/${encodeURIComponent(voiceId)}/samples/${index}?delete_audio=${deleteAudio}`,
      { method: "POST" },
    ),
  rename: (voiceId: string, displayName: string) =>
    json<{ ok: boolean }>(`/api/identities/${encodeURIComponent(voiceId)}/rename`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ display_name: displayName }),
    }),
};
