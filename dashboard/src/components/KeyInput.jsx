import React, { useState, useEffect } from 'react';
import { Eye, EyeOff, Check, Sparkles, Loader2, AlertCircle, ChevronDown } from 'lucide-react';
import { apiFetch } from '../lib/api';

// The backend reports prices per token; the picker shows the friendlier per-1M figure.
function perMillion(price) {
  const n = Number(price);
  if (!Number.isFinite(n) || n <= 0) return null;
  return `$${(n * 1000000).toPrecision(3)}/1M`;
}

export default function KeyInput({
  // Gemini — the default provider for the AI Shorts steps.
  onKeySet,
  savedKey,
  // OpenRouter — an optional BYOK alternative. Runs the same three AI steps
  // (web research, analysis, scripts) with no Google key.
  provider = 'gemini',
  onProviderChange,
  openRouterKey = '',
  onOpenRouterKeySet,
  model = '',
  onModelChange,
}) {
  const [key, setKey] = useState(savedKey || '');
  const [isVisible, setIsVisible] = useState(false);
  const [isSaved, setIsSaved] = useState(!!savedKey);

  const [orKey, setOrKey] = useState(openRouterKey || '');
  const [orVisible, setOrVisible] = useState(false);
  const [orSaved, setOrSaved] = useState(!!openRouterKey);

  const [models, setModels] = useState([]);
  const [modelsLive, setModelsLive] = useState(false);
  const [modelsError, setModelsError] = useState('');
  const [loadingModels, setLoadingModels] = useState(false);

  useEffect(() => {
    if (savedKey) setKey(savedKey);
  }, [savedKey]);

  useEffect(() => {
    if (openRouterKey) setOrKey(openRouterKey);
  }, [openRouterKey]);

  // Load the qualifying OpenRouter catalogue once a key is available. Bound to
  // the *saved* key rather than the typing buffer, so a half-typed secret is
  // never sent and the dropdown refreshes exactly when the key in use changes.
  // The endpoint always returns the curated list too, so it is never empty.
  useEffect(() => {
    if (provider !== 'openrouter' || !openRouterKey.trim()) {
      setModels([]);
      setModelsLive(false);
      setModelsError('');
      return;
    }
    let cancelled = false;
    setLoadingModels(true);
    setModelsError('');
    apiFetch('/api/saasshorts/openrouter/models', {
      headers: { 'X-OpenRouter-Key': openRouterKey.trim() },
    })
      .then((r) => r.json())
      .then((data) => {
        if (cancelled) return;
        const list = data.models || [];
        setModels(list);
        setModelsLive(!!data.live);
        setModelsError(data.error || '');
        // Keep a valid *and selectable* model: if the stored model vanished, was
        // never set, or is no longer in the curated set, fall back to the first
        // curated entry rather than silently keeping something unselectable.
        const selectableList = list.filter((m) => m.recommended);
        if (!model || !selectableList.some((m) => m.id === model)) {
          const first = selectableList[0];
          if (first) onModelChange?.(first.id);
        }
      })
      .catch(() => {
        if (!cancelled) setModelsError('Could not reach OpenRouter for the model list.');
      })
      .finally(() => {
        if (!cancelled) setLoadingModels(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [provider, orKey]);

  const handleSave = () => {
    if (key.trim().length > 0) {
      onKeySet(key);
      setIsSaved(true);
    }
  };

  const handleSaveOr = () => {
    if (orKey.trim().length > 0) {
      onOpenRouterKeySet(orKey);
      setOrSaved(true);
    }
  };

  const selected = models.find((m) => m.id === model);
  // The curated models are the selectable options. Each one is metadata-qualified
  // — it advertises JSON output with enough context and output budget for the
  // three AI steps — which is enough to *run* it on the caller's own OpenRouter
  // key. It is not proof the model runs the pipeline well: a curated model that
  // has not been end-to-end tested stays selectable but is labelled "Not
  // end-to-end tested" in the dropdown and below. The `verified` flag is never
  // implied by membership — it is set only by recorded end-to-end evidence — so
  // an untested model is never shown as verified.
  const selectable = models.filter((m) => m.recommended);
  const excluded = models.filter((m) => !m.recommended);
  // The live-only long tail is not curated, so it stays as diagnostic metadata:
  // summarised as a count rather than a wall of IDs.
  const excludedLiveCount = excluded.length;

  return (
    <div className="card p-4 sm:p-6 mb-8 animate-fade">
      <div className="flex items-center gap-3 mb-4">
        <div className="p-2 bg-paper3 rounded-input text-brass">
          <Sparkles size={18} />
        </div>
        <h2 className="font-display lowercase text-lg text-ink">AI Provider</h2>
      </div>

      {/* Provider toggle: one provider drives all three AI Shorts steps. */}
      <div className="grid grid-cols-2 gap-3 mb-5">
        {[
          { id: 'gemini', label: 'Gemini', note: 'Google AI Studio · free tier' },
          { id: 'openrouter', label: 'OpenRouter', note: 'BYOK · 300+ models' },
        ].map((p) => (
          <button
            key={p.id}
            onClick={() => onProviderChange?.(p.id)}
            className={`card card-hover p-3 text-left transition-colors ${
              provider === p.id ? 'border-brass' : ''
            }`}
          >
            <div className="text-sm font-medium lowercase text-ink">{p.label}</div>
            <div className="text-xs text-muted">{p.note}</div>
          </button>
        ))}
      </div>

      {provider === 'gemini' ? (
        <>
          <div className="flex flex-col sm:flex-row gap-3">
            <div className="relative sm:flex-1">
              <input
                type={isVisible ? 'text' : 'password'}
                value={key}
                onChange={(e) => {
                  setKey(e.target.value);
                  setIsSaved(false);
                }}
                placeholder="AIzaSy..."
                className="input-field pr-12 font-mono"
              />
              <button
                onClick={() => setIsVisible(!isVisible)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-muted hover:text-ink transition-colors"
              >
                {isVisible ? <EyeOff size={18} /> : <Eye size={18} />}
              </button>
            </div>
            <button
              onClick={handleSave}
              disabled={!key || isSaved}
              className={isSaved ? 'badge-ok px-4 cursor-default' : 'btn-primary'}
            >
              {isSaved ? (
                <>
                  <Check size={14} /> Ready
                </>
              ) : (
                'Set Key'
              )}
            </button>
          </div>
          <p className="mt-3 text-xs text-muted">
            Your key is stored locally in your browser for convenience.
            <br />
            <a
              href="https://aistudio.google.com/app/apikey"
              target="_blank"
              rel="noopener noreferrer"
              className="text-brass hover:underline mt-1 inline-block"
            >
              Get your free Gemini API Key here →
            </a>
          </p>
        </>
      ) : (
        <>
          <div className="flex flex-col sm:flex-row gap-3">
            <div className="relative sm:flex-1">
              <input
                type={orVisible ? 'text' : 'password'}
                value={orKey}
                onChange={(e) => {
                  setOrKey(e.target.value);
                  setOrSaved(false);
                }}
                placeholder="sk-or-v1-..."
                className="input-field pr-12 font-mono"
              />
              <button
                onClick={() => setOrVisible(!orVisible)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-muted hover:text-ink transition-colors"
              >
                {orVisible ? <EyeOff size={18} /> : <Eye size={18} />}
              </button>
            </div>
            <button
              onClick={handleSaveOr}
              disabled={!orKey || orSaved}
              className={orSaved ? 'badge-ok px-4 cursor-default' : 'btn-primary'}
            >
              {orSaved ? (
                <>
                  <Check size={14} /> Ready
                </>
              ) : (
                'Set Key'
              )}
            </button>
          </div>

          {/* Single model dropdown for all three AI Shorts steps. */}
          <div className="mt-5">
            <label className="eyebrow block mb-2">Model</label>
            {loadingModels ? (
              <div className="text-xs text-muted flex items-center gap-2">
                <Loader2 size={14} className="animate-spin" /> Loading models…
              </div>
            ) : models.length === 0 ? (
              <p className="text-xs text-muted">
                {orKey.trim()
                  ? 'No qualifying models — every option needs JSON output support.'
                  : 'Set your OpenRouter key to load the model list.'}
              </p>
            ) : selectable.length === 0 ? (
              <p className="text-xs text-muted">
                No curated models available — the catalogue ships in-repo, so this
                should not happen; re-paste your OpenRouter key to reload it.
              </p>
            ) : (
              <div className="relative">
                <select
                  value={model}
                  onChange={(e) => onModelChange?.(e.target.value)}
                  className="input-field pr-10 appearance-none w-full"
                >
                  {selectable.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.name} — {m.id}
                      {m.free ? ' (free)' : ''}
                      {m.verified ? '' : ' · Not end-to-end tested'}
                    </option>
                  ))}
                </select>
                <ChevronDown
                  size={16}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted pointer-events-none"
                />
              </div>
            )}
            {excluded.length > 0 && (
              <p className="mt-2.5 text-xs text-muted">
                …and {excludedLiveCount} other qualifying OpenRouter model
                {excludedLiveCount === 1 ? '' : 's'} — metadata only, not curated
                and not end-to-end tested.
              </p>
            )}
            {selected && (
              <div className="mt-2.5 flex flex-wrap items-center gap-2 text-xs">
                {selected.free ? (
                  <span className="badge-ok">Free</span>
                ) : (
                  <span className="badge-brass">Paid</span>
                )}
                {selected.recommended && <span className="badge-ok">Recommended</span>}
                {selected.verified ? (
                  <span className="badge-ok">Verified end-to-end</span>
                ) : (
                  <span className="badge-warn">Not end-to-end tested</span>
                )}
                {perMillion(selected.prompt_price) && (
                  <span className="text-muted">
                    {perMillion(selected.prompt_price)} in ·{' '}
                    {perMillion(selected.completion_price)} out
                  </span>
                )}
                {selected.web_search_price != null ? (
                  <span className="text-muted">
                    + ${Number(selected.web_search_price).toFixed(3)} / web search
                  </span>
                ) : (
                  <span className="text-muted">web search billed by the provider</span>
                )}
              </div>
            )}
            {modelsError && (
              <p className="mt-2 text-xs text-warn flex items-start gap-1.5">
                <AlertCircle size={13} className="shrink-0 mt-0.5" />
                {modelsError} Showing the built-in curated list.
              </p>
            )}
            {!modelsLive && models.length > 0 && !modelsError && (
              <p className="mt-2 text-xs text-muted">
                Showing the built-in list — these are frozen snapshots of name and price.
              </p>
            )}
          </div>

          <p className="mt-3 text-xs text-muted">
            Your key is stored locally in your browser and only sent to process your request.
            <br />
            <a
              href="https://openrouter.ai/keys"
              target="_blank"
              rel="noopener noreferrer"
              className="text-brass hover:underline mt-1 inline-block"
            >
              Get an OpenRouter API key here →
            </a>
          </p>
        </>
      )}
    </div>
  );
}
