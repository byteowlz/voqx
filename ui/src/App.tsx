import { useEffect, useMemo, useState } from "react";
import {
  applyScheme,
  builtInSchemeList,
  type Scheme,
} from "@byteowlz/design-system";
import {
  api,
  type AnalysisResponse,
  type IdentityAnalysis,
  type IdentitySummary,
  type RegistryResponse,
} from "./api.js";
import { IdentityDetail } from "./IdentityDetail.js";
import { SimilarityMatrix } from "./SimilarityMatrix.js";

export default function App() {
  const schemes = builtInSchemeList as Scheme[];
  const [schemeId, setSchemeId] = useState(schemes[0]?.id ?? "");
  const [registry, setRegistry] = useState<RegistryResponse | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [filter, setFilter] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const scheme = schemes.find((s) => s.id === schemeId);
    if (scheme) applyScheme(scheme);
  }, [schemeId, schemes]);

  const reload = useMemo(
    () => async () => {
      try {
        const [reg, ana] = await Promise.all([api.registry(), api.analysis()]);
        setRegistry(reg);
        setAnalysis(ana);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      }
    },
    [],
  );

  useEffect(() => {
    void reload();
  }, [reload]);

  const analysisById = useMemo(() => {
    const map = new Map<string, IdentityAnalysis>();
    for (const item of analysis?.analyses ?? []) map.set(item.voice_id, item);
    return map;
  }, [analysis]);

  const suspiciousById = useMemo(() => {
    const map = new Map<string, number>();
    for (const item of analysis?.analyses ?? []) {
      map.set(item.voice_id, item.samples.filter((s) => s.misattributed || s.weak).length);
    }
    return map;
  }, [analysis]);

  const identities = registry?.identities ?? [];
  const visible = identities.filter((identity) => {
    const needle = filter.trim().toLowerCase();
    if (!needle) return true;
    return (
      identity.display_name.toLowerCase().includes(needle) ||
      identity.voice_id.toLowerCase().includes(needle)
    );
  });

  const current = identities.find((i) => i.voice_id === selected) ?? null;
  const totalSuspicious = [...suspiciousById.values()].reduce((a, b) => a + b, 0);

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <h1>voqx</h1>
          <span className="muted">voice registry</span>
        </div>
        <div className="root-path">{registry?.root ?? "connecting…"}</div>

        <select
          className="scheme"
          value={schemeId}
          onChange={(e) => setSchemeId(e.target.value)}
          style={{ width: "100%", marginBottom: "0.75rem" }}
        >
          {schemes.map((scheme) => (
            <option key={scheme.id} value={scheme.id}>
              {scheme.name ?? scheme.id}
            </option>
          ))}
        </select>

        <input
          className="filter"
          placeholder="Filter identities…"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
        />

        <button
          type="button"
          className="identity-item"
          aria-current={selected === null}
          onClick={() => setSelected(null)}
        >
          <span className="name">Overview</span>
          {totalSuspicious > 0 && <span className="badge alert">{totalSuspicious}</span>}
        </button>

        <ul className="identity-list">
          {visible.map((identity) => (
            <li key={identity.voice_id}>
              <IdentityRow
                identity={identity}
                selected={identity.voice_id === selected}
                suspicious={suspiciousById.get(identity.voice_id) ?? 0}
                onSelect={() => setSelected(identity.voice_id)}
              />
            </li>
          ))}
        </ul>
      </aside>

      <main className="main">
        {error && <div className="error">{error}</div>}
        {current ? (
          <IdentityDetail
            identity={current}
            analysis={analysisById.get(current.voice_id) ?? null}
            onChanged={reload}
            onError={setError}
          />
        ) : (
          <Overview analysis={analysis} onSelect={setSelected} />
        )}
      </main>
    </div>
  );
}

function IdentityRow({
  identity,
  selected,
  suspicious,
  onSelect,
}: {
  identity: IdentitySummary;
  selected: boolean;
  suspicious: number;
  onSelect: () => void;
}) {
  return (
    <button type="button" className="identity-item" aria-current={selected} onClick={onSelect}>
      <span className="name">{identity.display_name}</span>
      {identity.recognition && <span className="badge rec">R</span>}
      {identity.synthesis && <span className="badge syn">S</span>}
      {identity.wake && <span className="badge wake">W</span>}
      {suspicious > 0 && <span className="badge alert">{suspicious}</span>}
    </button>
  );
}

function Overview({
  analysis,
  onSelect,
}: {
  analysis: AnalysisResponse | null;
  onSelect: (id: string) => void;
}) {
  if (!analysis) return <p className="empty">Loading analysis…</p>;

  const ranked = [...analysis.analyses].sort((a, b) => {
    const sa = a.samples.filter((s) => s.misattributed || s.weak).length;
    const sb = b.samples.filter((s) => s.misattributed || s.weak).length;
    if (sa !== sb) return sb - sa;
    return (a.mean_leave_one_out ?? 1) - (b.mean_leave_one_out ?? 1);
  });

  return (
    <>
      <header>
        <h2>Enrollment quality</h2>
        <span className="voice-id">
          engine {analysis.engine} · leave-one-out cosine · evidence only
        </span>
      </header>

      <section className="section">
        <h3>Identities by risk</h3>
        <table>
          <thead>
            <tr>
              <th>Identity</th>
              <th>Samples</th>
              <th>Mean leave-one-out</th>
              <th>Flagged</th>
            </tr>
          </thead>
          <tbody>
            {ranked.map((item) => {
              const flagged = item.samples.filter((s) => s.misattributed || s.weak).length;
              return (
                <tr key={item.voice_id} className={flagged > 0 ? "flagged" : undefined}>
                  <td>
                    <button
                      type="button"
                      className="action"
                      onClick={() => onSelect(item.voice_id)}
                    >
                      {item.display_name}
                    </button>
                  </td>
                  <td className="num">{item.samples.length}</td>
                  <td>
                    <ScoreBar value={item.mean_leave_one_out} />
                  </td>
                  <td>{flagged > 0 ? <span className="badge alert">{flagged}</span> : "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>

      <section className="section">
        <h3>Centroid similarity</h3>
        <p className="muted" style={{ marginTop: "-0.4rem", fontSize: "0.78rem" }}>
          High off-diagonal values mean two identities are hard to tell apart.
        </p>
        <SimilarityMatrix matrix={analysis.matrix} />
      </section>
    </>
  );
}

export function ScoreBar({ value }: { value: number | null }) {
  if (value === null) {
    return <span className="muted">—</span>;
  }
  const pct = Math.max(0, Math.min(1, value)) * 100;
  const tone = value >= 0.6 ? "" : value >= 0.45 ? "warn" : "bad";
  return (
    <span className="score">
      <span className="bar">
        <span className={tone} style={{ width: `${pct}%` }} />
      </span>
      <span className="num">{value.toFixed(3)}</span>
    </span>
  );
}
