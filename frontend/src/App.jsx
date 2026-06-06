import React, { useState, useEffect, useRef } from 'react';
import { 
  Activity, 
  Map, 
  BarChart2, 
  CheckSquare, 
  Download, 
  Upload, 
  File, 
  Layers, 
  Sliders, 
  ChevronRight, 
  RefreshCw,
  X
} from 'lucide-react';

import MapView from './components/MapView';
import ChartsView from './components/ChartsView';
import ReviewQueue from './components/ReviewQueue';
import GalleryModal from './components/GalleryModal';

export default function App() {
  // Navigation & UI state
  const [activeTab, setActiveTab] = useState('map');
  const [galleryOpen, setGalleryOpen] = useState(false);
  const [busy, setBusy] = useState(false);
  const [dragOver, setDragOver] = useState(false);

  // Model & Server Status
  const [serverStatus, setServerStatus] = useState('loading'); // 'ok', 'error', 'loading'
  const [statusMessage, setStatusMessage] = useState('Connecting...');
  const [allSpecies, setAllSpecies] = useState([]);
  const [speciesColors, setSpeciesColors] = useState({});

  // User input selection state
  const [selectedFile, setSelectedFile] = useState(null); // Local File uploaded
  const [selectedName, setSelectedName] = useState(null); // File name from server gallery
  const [selectedThumb, setSelectedThumb] = useState(''); // Selected server thumbnail url
  const [selectedSizeMb, setSelectedSizeMb] = useState('');
  const [imageUrl, setImageUrl] = useState('');           // Full-res image for detection view
  const [isAnnotated, setIsAnnotated] = useState(false);  // true after YOLO-annotated image loaded

  // Sliders state
  const [confAuto, setConfAuto] = useState(0.80);
  const [confMin, setConfMin] = useState(0.35);

  // Filters & predictions data state
  const [speciesFilter, setSpeciesFilter] = useState({}); // { Mango: true, Neem: true ... }
  const [predictionData, setPredictionData] = useState(null);
  const [exportLogs, setExportLogs] = useState([]);

  const fileInputRef = useRef(null);

  // 1. Handshake status with server on mount
  useEffect(() => {
    checkServerStatus();
  }, []);

  const checkServerStatus = async () => {
    setServerStatus('loading');
    setStatusMessage('Connecting...');
    try {
      const res = await fetch('/api/status');
      if (!res.ok) throw new Error('Status endpoint failed');
      const data = await res.json();

      const species = data.species || [];
      // Always include 'Other' for uncertain/overlapping detections
      if (!species.includes('Other')) species.push('Other');

      const colors = data.species_colors || {};
      colors['Other'] = colors['Other'] || '#94A3B8';

      setAllSpecies(species);
      setSpeciesColors(colors);
      setConfMin(data.conf_min || 0.35);
      setConfAuto(data.conf_auto || 0.80);

      // Create filter object (all checked by default)
      const initialFilters = {};
      species.forEach((sp) => {
        initialFilters[sp] = true;
      });
      setSpeciesFilter(initialFilters);


      if (data.model_loaded) {
        setServerStatus('ok');
        setStatusMessage('Model ready');
      } else {
        setServerStatus('error');
        setStatusMessage('Model not loaded');
      }
    } catch (err) {
      setServerStatus('error');
      setStatusMessage('Server offline');
      console.error('[app] Status check failed:', err);
    }
  };

  // ── Drag & Drop Handlers ──
  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = () => {
    setDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) {
      selectUploadedFile(file);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      selectUploadedFile(file);
    }
  };

  const selectUploadedFile = (file) => {
    setSelectedFile(file);
    setSelectedName(null);
    // Generate a local object URL so the sidebar and detection view can show the image
    const objectUrl = URL.createObjectURL(file);
    setSelectedThumb(objectUrl);
    setImageUrl(objectUrl);  // Use same object URL for full-res detection overlay
    setSelectedSizeMb((file.size / 1048576).toFixed(1) + ' MB');
  };

  const selectGalleryImage = (name) => {
    setSelectedName(name);
    setSelectedFile(null);
    setSelectedThumb(`/api/images/${encodeURIComponent(name)}/thumb`);
    setImageUrl(`/api/images/${encodeURIComponent(name)}/full`); // Full-res for detection overlay
    setSelectedSizeMb('Server tile');
  };

  const clearSelection = () => {
    setSelectedFile(null);
    setSelectedName(null);
    setSelectedThumb('');
    setImageUrl('');
    setIsAnnotated(false);
    setSelectedSizeMb('');
  };

  // ── Slider Collision Rules ──
  const handleConfAutoChange = (val) => {
    setConfAuto(val);
    if (val < confMin) {
      setConfMin(val);
    }
  };

  const handleConfMinChange = (val) => {
    setConfMin(val);
    if (val > confAuto) {
      setConfAuto(val);
    }
  };

  // ── Run Prediction ──
  const handleRunPrediction = async () => {
    if (!selectedName && !selectedFile) return;
    setBusy(true);

    try {
      let data = null;
      if (selectedName) {
        // Path A: Predict server-side image by filename
        const res = await fetch('/api/predict-by-name', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ filename: selectedName, conf_min: confMin }),
        });
        if (!res.ok) throw new Error((await res.json()).detail || 'Prediction failed');
        data = await res.json();
      } else if (selectedFile) {
        // Path B: Predict by raw multipart upload
        const form = new FormData();
        form.append('file', selectedFile);
        form.append('conf_min', confMin);

        const res = await fetch('/api/predict', {
          method: 'POST',
          body: form,
        });
        if (!res.ok) throw new Error((await res.json()).detail || 'Prediction failed');
        data = await res.json();
      }

      if (data) {
        // Re-classify features using the frontend's confAuto slider
        data.features = (data.features || []).map((feat) => {
          const status = feat.confidence >= confAuto ? 'AUTO_ACCEPTED' : 'REVIEW_REQUIRED';
          return { ...feat, status };
        });
        data.summary = calculateSummary(data.features);

        // ── Fetch the YOLO-annotated image (boxes drawn natively by the model)
        // This guarantees boxes are positioned exactly where YOLO detected
        try {
          let annotatedRes;
          if (selectedName) {
            annotatedRes = await fetch('/api/annotated-image/by-name', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({ filename: selectedName, conf_min: confMin }),
            });
          } else if (selectedFile) {
            const form = new FormData();
            form.append('file', selectedFile);
            form.append('conf_min', confMin);
            annotatedRes = await fetch('/api/annotated-image/upload', {
              method: 'POST',
              body: form,
            });
          }
          if (annotatedRes && annotatedRes.ok) {
            const blob = await annotatedRes.blob();
            const annotatedUrl = URL.createObjectURL(blob);
            setImageUrl(annotatedUrl);   // replace raw image with annotated result
            setIsAnnotated(true);        // tell MapView boxes are already baked in
          }
        } catch (annErr) {
          console.warn('[app] Annotated image fetch failed, keeping raw image:', annErr);
        }

        setPredictionData(data);
        logOutputs(data);
        setActiveTab('map');
      }

    } catch (err) {
      alert(`Prediction Failed: ${err.message}`);
    } finally {
      setBusy(false);
    }
  };

  // Helper summary recalculator
  const calculateSummary = (features) => {
    const total = features.length;
    const auto_accepted = features.filter((f) => f.status === 'AUTO_ACCEPTED').length;
    const review_required = features.filter((f) => f.status === 'REVIEW_REQUIRED').length;
    
    let sumConf = 0;
    const species_counts = {};

    features.forEach((f) => {
      sumConf += f.confidence;
      species_counts[f.species] = (species_counts[f.species] || 0) + 1;
    });

    const avg_confidence = total > 0 ? sumConf / total : 0;

    return {
      total,
      auto_accepted,
      review_required,
      avg_confidence,
      species_counts,
    };
  };

  // Log compiled files
  const logOutputs = (data) => {
    const time = new Date().toLocaleTimeString();
    const logs = [];
    if (data.saved_csv) logs.push({ time, path: data.saved_csv, type: 'CSV' });
    if (data.saved_gpkg) logs.push({ time, path: data.saved_gpkg, type: 'GPKG' });
    setExportLogs((prev) => [...logs, ...prev]);
  };

  // ── Human validation callback ──
  const handleQueueResolve = (id, correctedSpecies) => {
    if (!predictionData) return;

    // Mutate the local master prediction state
    const updatedFeatures = predictionData.features.map((feat) => {
      if (feat.id === id) {
        return {
          ...feat,
          species: correctedSpecies,
          status: 'MANUALLY_VERIFIED',
          color: speciesColors[correctedSpecies] || '#94A3B8',
        };
      }
      return feat;
    });

    const updatedSummary = calculateSummary(updatedFeatures);

    setPredictionData((prev) => ({
      ...prev,
      features: updatedFeatures,
      summary: updatedSummary,
    }));
  };

  // ── Species checkbox toggle handlers ──
  const handleSpeciesFilterChange = (species, checked) => {
    setSpeciesFilter((prev) => ({
      ...prev,
      [species]: checked,
    }));
  };

  // ── Data Export Stream handshakes ──
  const handleExportCSV = async (type = 'raw') => {
    if (!predictionData) return;

    let featuresToExport = predictionData.features;
    let filePrefix = 'raw_';

    if (type === 'validated') {
      featuresToExport = predictionData.features.filter(
        (f) => f.status !== 'REVIEW_REQUIRED'
      );
      filePrefix = 'validated_';
    }

    try {
      const res = await fetch('/api/export/csv', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          features: featuresToExport,
          filename: predictionData.filename.replace(/\.[^/.]+$/, ''),
        }),
      });

      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${filePrefix}${predictionData.filename.replace(/\.[^/.]+$/, '')}.csv`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err) {
      console.error('[export] CSV fail:', err);
    }
  };

  const handleExportGPKG = () => {
    if (!predictionData || !predictionData.saved_gpkg) return;
    alert(
      `GeoPackage compiled & exported:\n\n${predictionData.saved_gpkg}\n\nYou can load this directly into QGIS, ArcGIS, or other spatial tools.`
    );
  };

  // ── Stat calculations ──
  const statTotal = predictionData?.summary?.total ?? '—';
  const statAuto = predictionData?.summary?.auto_accepted ?? '—';
  const statReview = predictionData?.summary?.review_required ?? '—';
  const statConf = predictionData?.summary?.avg_confidence 
    ? (predictionData.summary.avg_confidence * 100).toFixed(0) + '%' 
    : '—';
  const statCountSpecies = predictionData?.summary?.species_counts 
    ? Object.keys(predictionData.summary.species_counts).length 
    : '—';

  return (
    <div className="app-container">
      
      {/* ── HEADER NAVIGATION ── */}
      <nav className="nav">
        <div className="nav-brand">
          <span className="nav-icon">🌳</span>
          <span className="nav-title">TreeVision</span>
          <span className="nav-version">v2.0</span>
        </div>

        <div className="nav-tabs">
          <button 
            className={`tab ${activeTab === 'map' ? 'active' : ''}`}
            onClick={() => setActiveTab('map')}
          >
            <Map size={16} />
            <span>Map Grid</span>
          </button>
          
          <button 
            className={`tab ${activeTab === 'charts' ? 'active' : ''}`}
            onClick={() => setActiveTab('charts')}
          >
            <BarChart2 size={16} />
            <span>Telemetry</span>
          </button>
          
          <button 
            className={`tab ${activeTab === 'queue' ? 'active' : ''}`}
            onClick={() => setActiveTab('queue')}
          >
            <CheckSquare size={16} />
            <span>Review</span>
            {statReview !== '—' && statReview > 0 && (
              <span className="queue-badge">{statReview}</span>
            )}
          </button>
          
          <button 
            className={`tab ${activeTab === 'export' ? 'active' : ''}`}
            onClick={() => setActiveTab('export')}
          >
            <Download size={16} />
            <span>Export</span>
          </button>
        </div>

        <div className="nav-status">
          <span className={`status-dot ${serverStatus}`}></span>
          <span className="status-label">{statusMessage}</span>
          <button 
            onClick={checkServerStatus} 
            title="Refresh status connection"
            style={{ display: 'flex', marginLeft: '6px', cursor: 'pointer', color: 'var(--text-muted)' }}
          >
            <RefreshCw size={11} />
          </button>
        </div>
      </nav>

      {/* ── TELEMETRY STAT CARDS ── */}
      <div className="stat-strip">
        <div className="stat-card">
          <div className="stat-value">{statTotal}</div>
          <div className="stat-label">Canopies Detected</div>
        </div>
        <div className="stat-card accent-green">
          <div className="stat-value">{statAuto}</div>
          <div className="stat-label">Auto-Accepted</div>
        </div>
        <div className="stat-card accent-amber">
          <div className="stat-value">{statReview}</div>
          <div className="stat-label">Review Pending</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{statConf}</div>
          <div className="stat-label">Avg Confidence</div>
        </div>
        <div className="stat-card">
          <div className="stat-value">{statCountSpecies}</div>
          <div className="stat-label">Species Count</div>
        </div>
      </div>

      {/* ── CORE APPLICATION LAYOUT ── */}
      <div className="layout">
        
        {/* ── SIDEBAR CONTROLS ── */}
        <aside className="sidebar">
          
          <div className="sidebar-section">
            <div className="sidebar-label">Orthomosaic Source</div>

            {/* Selected image card */}
            {(selectedName || selectedFile) ? (
              <div className="selected-chip">
                {selectedThumb ? (
                  <img className="selected-thumb" src={selectedThumb} alt="thumbnail" />
                ) : (
                  <div className="selected-thumb" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#1a1e1b' }}>
                    📦
                  </div>
                )}
                <div className="selected-info">
                  <div className="selected-name">
                    {selectedName || selectedFile.name}
                  </div>
                  <div className="selected-meta">{selectedSizeMb}</div>
                </div>
                <button 
                  className="selected-clear" 
                  onClick={clearSelection}
                  title="Remove selected orthomosaic"
                >
                  <X size={14} />
                </button>
              </div>
            ) : (
              <>
                <button 
                  className="gallery-open-btn"
                  onClick={() => setGalleryOpen(true)}
                >
                  🖼️ Browse local image library
                </button>

                <div className="or-divider">
                  <span>or</span>
                </div>

                <div 
                  className={`upload-zone ${dragOver ? 'dragover' : ''}`}
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current.click()}
                >
                  <Upload className="upload-icon" />
                  <div className="upload-text">Drag & drop orthomosaic</div>
                  <div className="upload-sub">TIFF, JPEG, or PNG files</div>
                  <input 
                    type="file" 
                    ref={fileInputRef}
                    onChange={handleFileChange}
                    accept=".tif,.tiff,.jpg,.jpeg,.png"
                    hidden 
                  />
                </div>
              </>
            )}
          </div>

          <div className="sidebar-section">
            <div className="sidebar-label">Confidence Gates</div>
            
            <div className="slider-row">
              <span>Auto-Accept Lock</span>
              <span className="slider-val">{(confAuto * 100).toFixed(0)}%</span>
            </div>
            <input 
              type="range"
              className="slider"
              min="0"
              max="1"
              step="0.01"
              value={confAuto}
              onChange={(e) => handleConfAutoChange(parseFloat(e.target.value))}
            />

            <div className="slider-row" style={{ marginTop: '6px' }}>
              <span>Minimum Gate</span>
              <span className="slider-val">{(confMin * 100).toFixed(0)}%</span>
            </div>
            <input 
              type="range"
              className="slider"
              min="0"
              max="1"
              step="0.01"
              value={confMin}
              onChange={(e) => handleConfMinChange(parseFloat(e.target.value))}
            />
          </div>

          {allSpecies.length > 0 && (
            <div className="sidebar-section">
              <div className="sidebar-label">Layer Filter</div>
              <div className="species-filters">
                {allSpecies.map((sp) => {
                  const color = speciesColors[sp] || '#94A3B8';
                  return (
                    <label key={sp} className="species-filter-row">
                      <input 
                        type="checkbox"
                        checked={speciesFilter[sp] !== false}
                        onChange={(e) => handleSpeciesFilterChange(sp, e.target.checked)}
                      />
                      <span className="species-dot" style={{ backgroundColor: color }}></span>
                      <span>{sp}</span>
                    </label>
                  );
                })}
              </div>
            </div>
          )}

          <button 
            className="run-btn"
            disabled={(!selectedName && !selectedFile) || serverStatus !== 'ok' || busy}
            onClick={handleRunPrediction}
          >
            {busy ? (
              <>
                <span className="spinner"></span>
                <span>Calculating Canopy...</span>
              </>
            ) : (
              <>
                <span>Run Forest Scan</span>
                <ChevronRight size={18} />
              </>
            )}
          </button>
        </aside>

        {/* ── MAIN WORKSPACE CONTENT PANELS ── */}
        <main className="content">
          
          {/* MAP LAYER */}
          <div className={`panel ${activeTab === 'map' ? 'active' : ''}`}>
            <MapView 
              data={predictionData} 
              speciesFilter={speciesFilter}
              speciesColors={speciesColors}
              imageUrl={imageUrl}
              isAnnotated={isAnnotated}
            />
          </div>

          {/* TELEMETRY CHARTS LAYER */}
          <div className={`panel ${activeTab === 'charts' ? 'active' : ''}`}>
            <ChartsView 
              data={predictionData} 
              speciesColors={speciesColors}
            />
          </div>

          {/* MANUAL VALIDATION LAYER */}
          <div className={`panel ${activeTab === 'queue' ? 'active' : ''}`}>
            <ReviewQueue 
              data={predictionData}
              allSpecies={allSpecies}
              speciesColors={speciesColors}
              onResolve={handleQueueResolve}
            />
          </div>

          {/* EXPORT OPTIONS LAYER */}
          <div className={`panel ${activeTab === 'export' ? 'active' : ''}`}>
            {predictionData ? (
              <div className="export-panel">
                <div className="export-grid">
                  <div className="export-card">
                    <div className="export-icon">📄</div>
                    <div className="export-name">Raw Telemetry (CSV)</div>
                    <div className="export-desc">
                      Export full unvalidated scans containing crowns, sizes, and confidence levels.
                    </div>
                    <button className="export-btn" onClick={() => handleExportCSV('raw')}>
                      Download Raw CSV
                    </button>
                  </div>

                  <div className="export-card">
                    <div className="export-icon">🗺️</div>
                    <div className="export-name">Crown Shapes (GeoPackage)</div>
                    <div className="export-desc">
                      Spatial GPKG format in EPSG:4326. Opens immediately in QGIS, ArcGIS.
                    </div>
                    <button 
                      className="export-btn" 
                      onClick={handleExportGPKG}
                      disabled={!predictionData.saved_gpkg}
                    >
                      Export Shapes
                    </button>
                  </div>

                  <div className="export-card">
                    <div className="export-icon">✅</div>
                    <div className="export-name">Validated Inventory (CSV)</div>
                    <div className="export-desc">
                      Exports final inventories, locking in auto-accepted and manual corrected labels.
                    </div>
                    <button className="export-btn" onClick={() => handleExportCSV('validated')}>
                      Download Inventory
                    </button>
                  </div>
                </div>

                {exportLogs.length > 0 && (
                  <div className="export-log">
                    <div className="log-title">Automated Server Storage Log</div>
                    <div className="log-items">
                      {exportLogs.map((log, i) => (
                        <div key={i} className="log-item">
                          <span>{log.time}</span>
                          <span style={{ color: 'var(--text-muted)' }}>[{log.type}]</span>
                          <span style={{ fontSize: '11px', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>
                            {log.path}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="queue-empty">
                <Download size={32} style={{ color: 'var(--text-muted)' }} />
                <span>Run prediction scans to enable file compilation exports.</span>
              </div>
            )}
          </div>
        </main>
      </div>

      {/* ── IMAGE GALLERY DIALOG MODAL ── */}
      <GalleryModal 
        isOpen={galleryOpen}
        onClose={() => setGalleryOpen(false)}
        onSelect={selectGalleryImage}
        selectedName={selectedName}
      />

    </div>
  );
}
