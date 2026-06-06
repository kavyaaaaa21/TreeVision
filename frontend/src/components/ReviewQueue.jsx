import React, { useState } from 'react';
import { Check, Edit2, AlertCircle } from 'lucide-react';

export default function ReviewQueue({ data, allSpecies, speciesColors, onResolve }) {
  const [resolvingIds, setResolvingIds] = useState({});
  const [selectedSpecies, setSelectedSpecies] = useState({});

  if (!data || !data.features) {
    return (
      <div className="queue-empty">
        <AlertCircle size={32} style={{ color: 'var(--text-muted)' }} />
        <span>Run a prediction to load the review queue.</span>
      </div>
    );
  }

  const reviewItems = data.features.filter((f) => f.status === 'REVIEW_REQUIRED');

  const handleSelectChange = (id, val) => {
    setSelectedSpecies((prev) => ({ ...prev, [id]: val }));
  };

  const handleValidate = async (id, originalSpecies, isCorrection = false) => {
    const feat = data.features.find((f) => f.id === id);
    if (!feat) return;

    const chosenSpecies = isCorrection 
      ? (selectedSpecies[id] || feat.species) 
      : feat.species;

    // 1. Trigger slide out animation in UI
    setResolvingIds((prev) => ({ ...prev, [id]: true }));

    // 2. Perform API call to register verification
    try {
      const res = await fetch('/api/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          id,
          corrected_species: chosenSpecies,
          original_species: originalSpecies,
          confidence: feat.confidence,
          filename: data.filename,
        }),
      });

      if (!res.ok) {
        console.warn('[queue] Validate failed:', await res.text());
      }
    } catch (err) {
      console.warn('[queue] Validate API call error:', err);
    }

    // 3. Complete animation transition and resolve state in parent
    setTimeout(() => {
      onResolve(id, chosenSpecies);
      setResolvingIds((prev) => {
        const copy = { ...prev };
        delete copy[id];
        return copy;
      });
      setSelectedSpecies((prev) => {
        const copy = { ...prev };
        delete copy[id];
        return copy;
      });
    }, 350);
  };

  return (
    <div className="queue-panel-wrap">
      <div className="queue-header">
        <div className="queue-title">Human-in-the-Loop Review Queue</div>
        <div className="queue-subtitle">
          {reviewItems.length > 0
            ? `${reviewItems.length} low-confidence detections require manual confirmation`
            : 'All detections were automatically accepted with high confidence! 🎉'}
        </div>
      </div>

      <div className="queue-list">
        {reviewItems.length === 0 ? (
          <div className="queue-empty">
            <Check size={40} style={{ color: 'var(--accent-green)', filter: 'drop-shadow(0 0 8px var(--accent-green-glow))' }} />
            <span style={{ fontSize: '15px', fontWeight: '500', color: 'var(--text-primary)' }}>
              All Detections Cleared & Verified!
            </span>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
              No review items remaining. You can export the final datasets.
            </span>
          </div>
        ) : (
          reviewItems.map((feat) => {
            const color = speciesColors[feat.species] || feat.color || '#94A3B8';
            const isResolving = resolvingIds[feat.id];
            const currentSelected = selectedSpecies[feat.id] || feat.species;

            return (
              <div
                key={feat.id}
                id={`qrow-${feat.id}`}
                className={`queue-item ${isResolving ? 'resolved' : ''}`}
                style={{ borderLeftColor: color }}
              >
                <div className="queue-thumb" style={{ borderColor: color + '22' }}>
                  🌿
                </div>
                <div className="queue-info">
                  <div className="queue-id">DETECTION ID #{feat.id}</div>
                  <div className="queue-species" style={{ color: color }}>
                    {feat.species}
                  </div>
                  <div className="queue-meta">
                    <span>Confidence: <strong>{(feat.confidence * 100).toFixed(1)}%</strong></span>
                    <span>·</span>
                    <span>Crown Area: <strong>{feat.crown_area_px.toLocaleString()} px²</strong></span>
                    <span>·</span>
                    <span className="badge review">Needs Review</span>
                  </div>
                </div>

                <div className="queue-actions">
                  <select
                    className="species-select"
                    value={currentSelected}
                    onChange={(e) => handleSelectChange(feat.id, e.target.value)}
                  >
                    {allSpecies.map((sp) => (
                      <option key={sp} value={sp}>
                        {sp}
                      </option>
                    ))}
                  </select>
                  <button
                    className="q-btn correct"
                    onClick={() => handleValidate(feat.id, feat.species, true)}
                    disabled={isResolving}
                  >
                    <Edit2 size={14} style={{ display: 'inline', marginRight: '4px', verticalAlign: 'middle' }} />
                    Correct
                  </button>
                  <button
                    className="q-btn accept"
                    onClick={() => handleValidate(feat.id, feat.species, false)}
                    disabled={isResolving}
                  >
                    <Check size={14} style={{ display: 'inline', marginRight: '4px', verticalAlign: 'middle' }} />
                    Accept
                  </button>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
