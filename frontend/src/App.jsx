import React, { useState, useEffect, useCallback } from "react";
import {
  Sparkles,
  Cpu,
  TrendingUp,
  Leaf,
  Activity,
  FileText,
  Layers,
  Database,
  Sliders,
  Copy,
  Check,
  Download,
  RefreshCw,
  Filter,
  AlertCircle,
  BarChart3,
  ExternalLink,
  Trash2,
  BookOpen,
  Upload,
  X,
  Wand2,
  Sun,
  Moon,
  ChevronDown,
  ChevronUp,
  Table
} from "lucide-react";

const API_URL = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/+$/, "");

const DEFAULT_PRESETS = {
  tech: {
    title: "AI & Cloud Tech",
    icon: "Cpu",
    text: `Artificial intelligence and machine learning models require large computational resources and high quality datasets for training. Deep neural networks perform representation learning on complex structured and unstructured data. Cloud infrastructure platforms like AWS, Google Cloud, and Azure allow software developers to deploy scalable AI microservices. Cybersecurity experts monitor global network traffic to protect enterprise cloud computing systems from security threats. Data engineering pipelines clean and aggregate telemetry before feeding data into predictive machine learning models. DevOps automation speeds up software delivery cycles across hybrid cloud server networks.`
  },
  economics: {
    title: "Global Finance & Economy",
    icon: "TrendingUp",
    text: `Central banks adjusted benchmark interest rates today to control rising inflation and stabilize currency valuations. Stock market indices experienced significant volatility as quarterly corporate earnings reports were released by financial institutions. International trade agreements and supply chain logistics continue to influence global commodity and consumer prices. Venture capital funds and angel investors are backing early stage technology startups despite higher borrowing costs. Macroeconomic policy balances employment targets against interest rate adjustments to foster long term monetary growth.`
  },
  climate: {
    title: "Climate & Renewable Energy",
    icon: "Leaf",
    text: `Renewable energy generation from commercial solar farms and offshore wind turbines reached unprecedented capacity levels this year. Climate scientists emphasize urgent reductions in carbon emissions to mitigate extreme weather phenomena worldwide. Oceanic temperatures are rising at historical rates, threatening delicate marine biodiversity and coastal coral reef ecosystems. Advanced grid battery storage technology is critical for balancing renewable power distribution across national electric grids.`
  },
  health: {
    title: "Healthcare & Genomics",
    icon: "Activity",
    text: `Medical researchers discovered a novel targeted immunotherapy treatment that selectively targets aggressive cancer cell receptors. Clinical trial phases demonstrated significant therapeutic efficacy and minimal adverse side effects compared to traditional chemotherapy. Next-generation genomic sequencing enables precision medicine tailored to individual patient genetic profiles and biomarkers. Diagnostic imaging software powered by computer vision assists radiologists in early detection of internal organ pathologies.`
  }
};

const DEFAULT_SAMPLE = DEFAULT_PRESETS.tech.text;

const TOPIC_COLORS = [
  { main: "#10b981", bg: "rgba(16, 185, 129, 0.12)", border: "rgba(16, 185, 129, 0.35)", name: "Teal" },
  { main: "#8b5cf6", bg: "rgba(139, 92, 246, 0.12)", border: "rgba(139, 92, 246, 0.35)", name: "Purple" },
  { main: "#f59e0b", bg: "rgba(245, 158, 11, 0.12)", border: "rgba(245, 158, 11, 0.35)", name: "Amber" },
  { main: "#f43f5e", bg: "rgba(244, 63, 94, 0.12)", border: "rgba(244, 63, 94, 0.35)", name: "Rose" },
  { main: "#06b6d4", bg: "rgba(6, 182, 212, 0.12)", border: "rgba(6, 182, 212, 0.35)", name: "Cyan" },
  { main: "#6366f1", bg: "rgba(99, 102, 241, 0.12)", border: "rgba(99, 102, 241, 0.35)", name: "Indigo" },
  { main: "#84cc16", bg: "rgba(132, 204, 22, 0.12)", border: "rgba(132, 204, 22, 0.35)", name: "Lime" },
  { main: "#ec4899", bg: "rgba(236, 72, 153, 0.12)", border: "rgba(236, 72, 153, 0.35)", name: "Pink" }
];

/** Confidence dot indicator — 5 dots filled proportionally to confidence */
function ConfidenceDots({ confidence, color }) {
  const filled = Math.round(confidence * 5);
  return (
    <span className="confidence-dots" title={`${(confidence * 100).toFixed(0)}% confidence`}>
      {Array.from({ length: 5 }, (_, i) => (
        <span
          key={i}
          className="conf-dot"
          style={{ backgroundColor: i < filled ? color : "rgba(255,255,255,0.12)" }}
        />
      ))}
    </span>
  );
}

/** "How it works" collapsible info drawer */
function InfoDrawer() {
  const [open, setOpen] = useState(false);
  return (
    <div className={`info-drawer ${open ? "open" : ""}`}>
      <button
        type="button"
        className="info-drawer-toggle"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
      >
        <BookOpen size={14} />
        <span>How does topic modeling work?</span>
        {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>
      <div className="info-drawer-body">
        <div className="info-grid">
          <div className="info-block">
            <h4>🎲 LDA — Latent Dirichlet Allocation</h4>
            <p>
              LDA is a <strong>generative probabilistic model</strong>. It assumes each document is
              a mixture of topics, and each topic is a distribution over words. The model learns these
              distributions by iterating over the corpus until stable. Best for <em>longer documents</em> with
              rich vocabulary.
            </p>
          </div>
          <div className="info-block">
            <h4>🧮 NMF — Non-Negative Matrix Factorization</h4>
            <p>
              NMF decomposes a <strong>TF-IDF matrix</strong> into two non-negative matrices: document-topic
              weights and topic-word weights. It tends to produce <em>more interpretable, crisper topics</em> and
              works well even on shorter texts.
            </p>
          </div>
          <div className="info-block">
            <h4>🔤 Bigrams</h4>
            <p>
              When enabled, the vectorizer considers two-word phrases (e.g. <em>"machine learning"</em>,{" "}
              <em>"climate change"</em>) as single features — capturing compound concepts that unigrams miss.
            </p>
          </div>
          <div className="info-block">
            <h4>📊 Reading the results</h4>
            <p>
              Each <strong>topic card</strong> shows the top words weighted by their relevance. The{" "}
              <strong>stacked bar</strong> shows how the document is proportionally split across topics.
              The <strong>segments panel</strong> shows sentence-level assignments with confidence scores.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function App() {
  const [text, setText] = useState(DEFAULT_SAMPLE);
  const [numTopics, setNumTopics] = useState(3);
  const [method, setMethod] = useState("lda");
  const [useBigrams, setUseBigrams] = useState(true);
  const [maxWords, setMaxWords] = useState(8);

  const [presets, setPresets] = useState(DEFAULT_PRESETS);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activePreset, setActivePreset] = useState("tech");
  const [selectedTopicFilter, setSelectedTopicFilter] = useState(null);
  const [copied, setCopied] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [suggesting, setSuggesting] = useState(false);
  const [suggestHint, setSuggestHint] = useState(null);
  const [exportingCsv, setExportingCsv] = useState(false);
  const [resultsKey, setResultsKey] = useState(0); // forces re-mount for animation

  // Theme: dark / light
  const [theme, setTheme] = useState(() => localStorage.getItem("nlp-theme") || "dark");

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("nlp-theme", theme);
  }, [theme]);

  // Fetch presets on load
  useEffect(() => {
    fetch(`${API_URL}/presets`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (data) setPresets(data);
      })
      .catch(() => {});
  }, []);

  // Keyboard shortcut: Ctrl+Enter
  useEffect(() => {
    const handler = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        handleSubmit();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  });

  const handlePresetSelect = (key) => {
    setActivePreset(key);
    setUploadedFile(null);
    setSuggestHint(null);
    if (presets && presets[key]) {
      setText(presets[key].text);
    } else if (DEFAULT_PRESETS[key]) {
      setText(DEFAULT_PRESETS[key].text);
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    setError(null);
    setResults(null);
    setSuggestHint(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_URL}/upload-file`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Failed to upload and parse file.");
      }

      setText(data.extracted_text);
      setUploadedFile({
        name: data.filename,
        size: data.size_bytes,
        type: data.content_type,
      });
      setActivePreset(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
      e.target.value = null;
    }
  };

  const handleAutoSuggest = async () => {
    if (!text.trim()) {
      setError("Please enter some text first to auto-detect topics.");
      return;
    }
    setSuggesting(true);
    setSuggestHint(null);
    try {
      const params = new URLSearchParams({ text: text.trim() });
      const res = await fetch(`${API_URL}/suggest-topics?${params}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Could not suggest topics.");
      setNumTopics(data.suggested_num_topics);
      setSuggestHint(data.reason);
    } catch (err) {
      setError(err.message);
    } finally {
      setSuggesting(false);
    }
  };

  const handleSubmit = useCallback(async (e) => {
    if (e) e.preventDefault();
    if (!text.trim()) {
      setError("Please paste or type text to analyze.");
      return;
    }

    setLoading(true);
    setError(null);
    setSelectedTopicFilter(null);
    setSuggestHint(null);

    try {
      const res = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          num_topics: Number(numTopics),
          method,
          use_bigrams: useBigrams,
          max_words_per_topic: Number(maxWords)
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "Failed to analyze topics.");
      }

      setResults(data);
      setResultsKey(k => k + 1); // bump key to trigger entrance animation
    } catch (err) {
      setError(
        err.message === "Failed to fetch"
          ? "Cannot connect to backend server. Please verify your FastAPI backend is running."
          : err.message
      );
    } finally {
      setLoading(false);
    }
  }, [text, numTopics, method, useBigrams, maxWords]);

  const handleCopySummary = () => {
    if (!results) return;
    const summaryText = results.topics
      .map(
        (t) =>
          `Topic ${t.id + 1} · ${t.label} (${t.percentage}%):\nTop Words: ` +
          t.top_words.map((w) => (typeof w === "object" ? `${w.word} (${(w.weight * 100).toFixed(1)}%)` : w)).join(", ")
      )
      .join("\n\n");

    navigator.clipboard.writeText(summaryText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleExportJSON = () => {
    if (!results) return;
    const jsonStr = JSON.stringify(results, null, 2);
    const blob = new Blob([jsonStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `topic-modeling-${method}-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleExportCSV = async () => {
    if (!results || exportingCsv) return;
    setExportingCsv(true);
    try {
      const res = await fetch(`${API_URL}/export-csv`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(results),
      });
      if (!res.ok) throw new Error("CSV export failed.");
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `topic-results-${method}-${new Date().toISOString().slice(0, 10)}.csv`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(err.message);
    } finally {
      setExportingCsv(false);
    }
  };

  const getPresetIcon = (iconName) => {
    switch (iconName) {
      case "Cpu": return <Cpu size={16} />;
      case "TrendingUp": return <TrendingUp size={16} />;
      case "Leaf": return <Leaf size={16} />;
      case "Activity": return <Activity size={16} />;
      default: return <BookOpen size={16} />;
    }
  };

  const filteredSegments = results?.segments
    ? selectedTopicFilter !== null
      ? results.segments.filter((s) => s.dominant_topic === selectedTopicFilter)
      : results.segments
    : [];

  return (
    <div className="app-container">
      {/* Glow ambient background elements */}
      <div className="glow-orb orb-1"></div>
      <div className="glow-orb orb-2"></div>

      <header className="header">
        <div className="header-top">
          <div className="badge-pill">
            <Sparkles size={14} className="sparkle-icon" />
            <span>NLP Intelligence Studio v2.1</span>
          </div>
          <button
            type="button"
            className="theme-toggle"
            onClick={() => setTheme(t => t === "dark" ? "light" : "dark")}
            title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
            aria-label="Toggle theme"
          >
            {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
          </button>
          <a
            href="https://github.com/adilurr"
            target="_blank"
            rel="noreferrer"
            className="github-link"
          >
            GitHub Repository <ExternalLink size={12} />
          </a>
        </div>

        <h1 className="header-title">Topic Modeling & Semantic Discovery</h1>
        <p className="header-subtitle">
          Uncover latent topics, term probability distributions, and sentence-level theme assignments using unsupervised <strong>LDA</strong> (Latent Dirichlet Allocation) or <strong>NMF</strong> (Non-Negative Matrix Factorization).
        </p>
      </header>

      {/* Main Control Panel */}
      <main className="main-content">
        <section className="glass-panel input-panel">
          {/* Input Panel Header with Presets & Upload */}
          <div className="input-panel-header">
            <div className="presets-bar">
              <span className="section-label">Quick Samples:</span>
              <div className="preset-buttons">
                {presets ? (
                  Object.entries(presets).map(([key, item]) => (
                    <button
                      key={key}
                      type="button"
                      className={`preset-btn ${activePreset === key ? "active" : ""}`}
                      onClick={() => handlePresetSelect(key)}
                    >
                      {getPresetIcon(item.icon)}
                      <span>{item.title}</span>
                    </button>
                  ))
                ) : (
                  <>
                    <button
                      type="button"
                      className={`preset-btn ${activePreset === "tech" ? "active" : ""}`}
                      onClick={() => { setActivePreset("tech"); setText(DEFAULT_SAMPLE); }}
                    >
                      <Cpu size={16} /> AI & Cloud Tech
                    </button>
                  </>
                )}
              </div>
            </div>

            <div className="upload-section">
               <label className={`upload-btn ${uploading ? "uploading" : ""}`}>
                  {uploading ? <RefreshCw size={16} className="spin-icon" /> : <Upload size={16} />}
                  <span>{uploading ? "Extracting..." : "Upload File"}</span>
                  <input
                    type="file"
                    accept=".txt,.md,.json,.csv,.tsv,.docx,.pdf"
                    onChange={handleFileUpload}
                    style={{ display: 'none' }}
                    disabled={uploading}
                  />
               </label>
            </div>
          </div>

          {/* Uploaded File Indicator */}
          {uploadedFile && (
            <div className="uploaded-file-card">
              <div className="file-info-group">
                <FileText size={18} className="file-icon" />
                <div className="file-details">
                  <span className="file-name">{uploadedFile.name}</span>
                  <span className="file-size">{(uploadedFile.size / 1024).toFixed(1)} KB extracted</span>
                </div>
              </div>
              <button
                className="remove-file-btn"
                onClick={() => { setUploadedFile(null); setText(""); }}
                title="Remove File"
              >
                <X size={16} />
              </button>
            </div>
          )}

          {/* Text Input Area */}
          <div className="input-group">
            <div className="textarea-wrapper">
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="Paste an article, paper abstract, notes, or multi-paragraph document here... (Ctrl+Enter to analyze)"
                rows={7}
              />
              <div className="textarea-footer">
                <span className="char-count">
                  {text.trim().split(/\s+/).filter(Boolean).length} words | {text.length} chars
                </span>
                {text && (
                  <button
                    type="button"
                    className="clear-btn"
                    onClick={() => { setText(""); setSuggestHint(null); }}
                    title="Clear text"
                  >
                    <Trash2 size={13} /> Clear
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Modeling Parameters Config Bar */}
          <div className="controls-grid">
            {/* Algorithm Selector */}
            <div className="control-item">
              <label>
                <Sliders size={14} /> Modeling Algorithm
              </label>
              <div className="toggle-group">
                <button
                  type="button"
                  className={`toggle-option ${method === "lda" ? "active" : ""}`}
                  onClick={() => setMethod("lda")}
                >
                  LDA <span className="toggle-sub">(Probabilistic)</span>
                </button>
                <button
                  type="button"
                  className={`toggle-option ${method === "nmf" ? "active" : ""}`}
                  onClick={() => setMethod("nmf")}
                >
                  NMF <span className="toggle-sub">(TF-IDF Vector)</span>
                </button>
              </div>
            </div>

            {/* Number of Topics */}
            <div className="control-item">
              <label htmlFor="numTopics">
                <BarChart3 size={14} /> Target Topics: <strong>{numTopics}</strong>
              </label>
              <div className="slider-wrapper">
                <input
                  id="numTopics"
                  type="range"
                  min={2}
                  max={8}
                  value={numTopics}
                  onChange={(e) => { setNumTopics(e.target.value); setSuggestHint(null); }}
                />
                <div className="slider-ticks">
                  <span>2</span>
                  <span>4</span>
                  <span>6</span>
                  <span>8</span>
                </div>
              </div>
            </div>

            {/* Max Words Per Topic */}
            <div className="control-item">
              <label htmlFor="maxWords">
                <Database size={14} /> Words / Topic: <strong>{maxWords}</strong>
              </label>
              <div className="slider-wrapper">
                <input
                  id="maxWords"
                  type="range"
                  min={3}
                  max={15}
                  value={maxWords}
                  onChange={(e) => setMaxWords(e.target.value)}
                />
                <div className="slider-ticks">
                  <span>3</span>
                  <span>8</span>
                  <span>15</span>
                </div>
              </div>
            </div>

            {/* Bigram Toggle */}
            <div className="control-item checkbox-item">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={useBigrams}
                  onChange={(e) => setUseBigrams(e.target.checked)}
                />
                <span>Include Bigrams (e.g. <em>"machine learning"</em>)</span>
              </label>
            </div>
          </div>

          {/* Auto-suggest hint */}
          {suggestHint && (
            <div className="suggest-hint">
              <Wand2 size={14} />
              <span>{suggestHint}</span>
            </div>
          )}

          {/* Info Drawer */}
          <InfoDrawer />

          {/* Submit Action Bar */}
          <div className="action-bar">
            <button
              type="button"
              className="suggest-btn"
              disabled={suggesting || !text.trim()}
              onClick={handleAutoSuggest}
              title="Auto-detect the ideal number of topics for this text"
            >
              {suggesting ? (
                <><RefreshCw size={15} className="spin-icon" /> Detecting...</>
              ) : (
                <><Wand2 size={15} /> Auto-detect Topics</>
              )}
            </button>
            <button
              type="button"
              className="submit-btn"
              disabled={loading}
              onClick={handleSubmit}
            >
              {loading ? (
                <>
                  <RefreshCw size={18} className="spin-icon" />
                  Extracting Topics...
                </>
              ) : (
                <>
                  <Sparkles size={18} />
                  Run Topic Analysis
                </>
              )}
            </button>
          </div>
        </section>

        {/* Error Alert */}
        {error && (
          <div className="error-banner">
            <AlertCircle size={20} />
            <span>{error}</span>
          </div>
        )}

        {/* Results Visualizer Dashboard */}
        {results && (
          <section className="results-container" key={resultsKey}>
            {/* Stats Metric Grid */}
            <div className="stats-grid">
              {[
                { icon: <FileText className="stat-icon" size={20} />, value: results.stats.total_words, label: "Total Words" },
                { icon: <Layers className="stat-icon" size={20} />, value: results.stats.total_sentences, label: "Segments Analyzed" },
                { icon: <Database className="stat-icon" size={20} />, value: results.stats.vocab_size, label: "Vocabulary Size" },
                { icon: <Sliders className="stat-icon" size={20} />, value: results.method_used, label: "Algorithm" },
              ].map((card, i) => (
                <div className="stat-card fade-slide-in" key={card.label} style={{ animationDelay: `${i * 60}ms` }}>
                  {card.icon}
                  <div className="stat-info">
                    <span className="stat-value">{card.value}</span>
                    <span className="stat-label">{card.label}</span>
                  </div>
                </div>
              ))}
            </div>

            {/* Document Composition Progress Bar */}
            <div className="glass-panel composition-panel fade-slide-in" style={{ animationDelay: "200ms" }}>
              <div className="panel-header">
                <h3>Overall Topic Composition</h3>
                <div className="export-actions">
                  <button type="button" className="action-pill-btn" onClick={handleCopySummary}>
                    {copied ? <Check size={14} /> : <Copy size={14} />}
                    {copied ? "Copied!" : "Copy Summary"}
                  </button>
                  <button type="button" className="action-pill-btn" onClick={handleExportJSON}>
                    <Download size={14} /> Export JSON
                  </button>
                  <button type="button" className="action-pill-btn" onClick={handleExportCSV} disabled={exportingCsv}>
                    {exportingCsv ? <RefreshCw size={14} className="spin-icon" /> : <Table size={14} />}
                    Export CSV
                  </button>
                </div>
              </div>

              <div className="stacked-progress-bar">
                {results.topics.map((t, idx) => {
                  const color = TOPIC_COLORS[idx % TOPIC_COLORS.length];
                  return (
                    <div
                      key={t.id}
                      className="progress-segment"
                      style={{ width: `${t.percentage}%`, backgroundColor: color.main }}
                      title={`Topic ${t.id + 1} · ${t.label}: ${t.percentage}%`}
                    />
                  );
                })}
              </div>

              <div className="composition-legend">
                {results.topics.map((t, idx) => {
                  const color = TOPIC_COLORS[idx % TOPIC_COLORS.length];
                  const isSelected = selectedTopicFilter === t.id;
                  return (
                    <button
                      key={t.id}
                      type="button"
                      className={`legend-item ${isSelected ? "active-legend" : ""}`}
                      onClick={() => setSelectedTopicFilter(isSelected ? null : t.id)}
                    >
                      <span className="legend-color-dot" style={{ backgroundColor: color.main }} />
                      <span className="legend-text">
                        T{t.id + 1} · {t.label} ({t.percentage}%)
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Topic Cards Grid */}
            <div className="topics-grid">
              {results.topics.map((topic, idx) => {
                const color = TOPIC_COLORS[idx % TOPIC_COLORS.length];
                const isSelected = selectedTopicFilter === topic.id;

                const weights = topic.top_words.map((w) =>
                  typeof w === "object" ? w.weight : 0.1
                );
                const maxWeight = Math.max(...weights, 0.01);

                return (
                  <div
                    key={topic.id}
                    className={`glass-panel topic-card fade-slide-in ${isSelected ? "selected-topic-card" : ""}`}
                    style={{
                      borderColor: isSelected ? color.main : "rgba(255, 255, 255, 0.08)",
                      animationDelay: `${260 + idx * 80}ms`
                    }}
                  >
                    <div className="topic-card-header">
                      <div className="topic-title-group">
                        <span
                          className="topic-badge-icon"
                          style={{ backgroundColor: color.bg, color: color.main, borderColor: color.border }}
                        >
                          T{topic.id + 1}
                        </span>
                        <div className="topic-title-stack">
                          <h4>Topic {topic.id + 1}</h4>
                          <span className="topic-label-tag" style={{ color: color.main }}>· {topic.label}</span>
                        </div>
                      </div>
                      <span
                        className="topic-percentage-chip"
                        style={{ color: color.main, backgroundColor: color.bg }}
                      >
                        {topic.percentage}% of doc
                      </span>
                    </div>

                    {/* Word Weight Visualizer */}
                    <div className="words-list">
                      {topic.top_words.map((item, wIdx) => {
                        const wordStr = typeof item === "object" ? item.word : item;
                        const weightVal = typeof item === "object" ? item.weight : 0.1;
                        const relPercent = (weightVal / maxWeight) * 100;

                        return (
                          <div key={wIdx} className="word-bar-row">
                            <div className="word-meta">
                              <span className="word-name">{wordStr}</span>
                              <span className="word-score">{(weightVal * 100).toFixed(1)}%</span>
                            </div>
                            <div className="word-bar-track">
                              <div
                                className="word-bar-fill"
                                style={{ width: `${relPercent}%`, backgroundColor: color.main }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Sentence-Level Segment Mapping */}
            <div className="glass-panel segments-panel fade-slide-in" style={{ animationDelay: "400ms" }}>
              <div className="segments-header">
                <div className="segments-title">
                  <Filter size={18} />
                  <h3>Sentence-Level Topic Mapping</h3>
                </div>

                <div className="filter-status">
                  {selectedTopicFilter !== null ? (
                    <span className="filter-chip">
                      Filtered: Topic {selectedTopicFilter + 1} · {results.topics[selectedTopicFilter]?.label}
                      <button type="button" onClick={() => setSelectedTopicFilter(null)}>×</button>
                    </span>
                  ) : (
                    <span className="filter-hint">
                      Click any topic above to filter matching sentences
                    </span>
                  )}
                </div>
              </div>

              <div className="segments-list">
                {filteredSegments.length > 0 ? (
                  filteredSegments.map((seg, si) => {
                    const color = TOPIC_COLORS[seg.dominant_topic % TOPIC_COLORS.length];
                    return (
                      <div
                        key={seg.id}
                        className="segment-card fade-slide-in"
                        style={{
                          borderLeftColor: color.main,
                          backgroundColor: color.bg,
                          animationDelay: `${si * 40}ms`
                        }}
                      >
                        <div className="segment-top">
                          <span
                            className="segment-topic-pill"
                            style={{ backgroundColor: color.main, color: "#0f172a" }}
                          >
                            Topic {seg.dominant_topic + 1} · {seg.topic_label}
                          </span>
                          <div className="segment-confidence-row">
                            <ConfidenceDots confidence={seg.confidence} color={color.main} />
                            <span className="confidence-tag">
                              {(seg.confidence * 100).toFixed(0)}%
                            </span>
                          </div>
                        </div>
                        <p className="segment-text">{seg.text}</p>
                      </div>
                    );
                  })
                ) : (
                  <p className="no-segments">No sentences matched the selected topic filter.</p>
                )}
              </div>
            </div>
          </section>
        )}
      </main>

      <footer className="footer">
        <p>
          Topic Intelligence Studio — Created with FastAPI, Scikit-Learn & React.
        </p>
        <div className="footer-links">
          <a href={`${API_URL}/docs`} target="_blank" rel="noreferrer">
            FastAPI Swagger Specs <ExternalLink size={12} />
          </a>
          <span>•</span>
          <a href="https://github.com/adilurr" target="_blank" rel="noreferrer">
            github.com/adilurr
          </a>
        </div>
      </footer>
    </div>
  );
}
