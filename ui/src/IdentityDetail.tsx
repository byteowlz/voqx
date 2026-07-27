import { useState } from "react";
import { api, type IdentityAnalysis, type IdentitySummary } from "./api.js";
import { ScoreBar } from "./App.js";

export function IdentityDetail({
  identity,
  analysis,
  onChanged,
  onError,
}: {
  identity: IdentitySummary;
  analysis: IdentityAnalysis | null;
  onChanged: () => Promise<void>;
  onError: (message: string | null) => void;
}) {
  const [busy, setBusy] = useState<number | null>(null);
  const [name, setName] = useState(identity.display_name);

  async function drop(index: number) {
    const confirmed = window.confirm(
      `Remove sample #${index + 1} from ${identity.display_name}?\n\n` +
        "The centroid is recomputed from the remaining samples. The audio file is kept.",
    );
    if (!confirmed) return;
    setBusy(index);
    try {
      await api.dropSample(identity.voice_id, index, false);
      onError(null);
      await onChanged();
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  }

  async function rename() {
    if (name.trim() === identity.display_name) return;
    try {
      await api.rename(identity.voice_id, name.trim());
      onError(null);
      await onChanged();
    } catch (err) {
      onError(err instanceof Error ? err.message : String(err));
    }
  }

  const flagged = analysis?.samples.filter((s) => s.misattributed || s.weak).length ?? 0;

  return (
    <>
      <header>
        <h2>{identity.display_name}</h2>
        <span className="voice-id">{identity.voice_id}</span>
        {identity.aliases.length > 0 && (
          <span className="voice-id"> · aka {identity.aliases.join(", ")}</span>
        )}
      </header>

      <section className="section">
        <div className="toolbar">
          <input
            className="filter"
            style={{ width: "220px", margin: 0 }}
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          <button
            type="button"
            className="action"
            disabled={name.trim() === identity.display_name || name.trim() === ""}
            onClick={() => void rename()}
          >
            Rename
          </button>
          <span className="muted" style={{ fontSize: "0.75rem" }}>
            the old name is kept as an alias
          </span>
        </div>

        <div className="toolbar">
          {identity.recognition && (
            <span className="badge rec">recognition · {identity.recognition_samples}</span>
          )}
          {identity.synthesis && (
            <span className="badge syn">synthesis · {identity.synthesis_references}</span>
          )}
          {identity.wake && <span className="badge wake">wake</span>}
          {flagged > 0 && <span className="badge alert">{flagged} flagged</span>}
        </div>
      </section>

      {analysis ? (
        <section className="section">
          <h3>Enrolled samples</h3>
          <p className="muted" style={{ marginTop: "-0.4rem", fontSize: "0.78rem" }}>
            Leave-one-out scores each sample against the centroid rebuilt without it — a
            contaminated sample shows a low score and matches another identity.
          </p>
          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Leave-one-out</th>
                <th>Best match</th>
                <th>Listen</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {analysis.samples.map((sample) => {
                const bad = sample.misattributed || sample.weak;
                return (
                  <tr key={sample.index} className={bad ? "flagged" : undefined}>
                    <td className="num">{sample.index + 1}</td>
                    <td style={{ minWidth: "160px" }}>
                      <ScoreBar value={sample.leave_one_out} />
                    </td>
                    <td>
                      {sample.misattributed ? (
                        <span className="badge alert">{sample.best_display_name}</span>
                      ) : (
                        <span className="muted">self</span>
                      )}{" "}
                      <span className="num muted">{sample.best_score.toFixed(3)}</span>
                    </td>
                    <td>
                      {sample.audio ? (
                        <audio controls preload="none" src={api.audioUrl(sample.audio)} />
                      ) : (
                        <span className="muted">no audio</span>
                      )}
                    </td>
                    <td style={{ textAlign: "right" }}>
                      <button
                        type="button"
                        className="action danger"
                        disabled={busy === sample.index}
                        onClick={() => void drop(sample.index)}
                      >
                        {busy === sample.index ? "removing…" : "Remove"}
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </section>
      ) : (
        <p className="empty">
          No recognition profile — nothing to analyse. This identity is synthesis-only.
        </p>
      )}
    </>
  );
}
