import type { AnalysisResponse } from "./api.js";

/** Heatmap of centroid similarity between identities. */
export function SimilarityMatrix({ matrix }: { matrix: AnalysisResponse["matrix"] }) {
  if (matrix.rows.length === 0) {
    return <p className="empty">No recognition profiles to compare.</p>;
  }
  return (
    <div style={{ overflowX: "auto" }}>
      <table className="matrix">
        <thead>
          <tr>
            <th className="rowhead" />
            {matrix.rows.map((row) => (
              <th key={row.voice_id} className="colhead">
                {row.display_name}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.rows.map((row) => (
            <tr key={row.voice_id}>
              <th className="rowhead">{row.display_name}</th>
              {row.scores.map((score, index) => {
                const self = matrix.voice_ids[index] === row.voice_id;
                return (
                  <td
                    key={matrix.voice_ids[index]}
                    className="cell"
                    style={{ background: cellColor(score, self), color: textColor(score, self) }}
                    title={`${row.display_name} ↔ ${matrix.rows[index]?.display_name}: ${score.toFixed(3)}`}
                  >
                    {score.toFixed(2)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Off-diagonal similarity is the risk signal, so only those get heat. */
function cellColor(score: number, self: boolean): string {
  if (self) return "color-mix(in srgb, var(--muted) 70%, transparent)";
  const clamped = Math.max(0, Math.min(1, score));
  return `color-mix(in srgb, var(--destructive) ${Math.round(clamped * 70)}%, transparent)`;
}

function textColor(score: number, self: boolean): string {
  if (self) return "var(--muted-foreground)";
  return score > 0.5 ? "var(--foreground)" : "var(--muted-foreground)";
}
