import React, { useState, useRef, useEffect, useCallback } from 'react';
import { ZoomIn, ZoomOut, Maximize2, Eye, EyeOff } from 'lucide-react';

export default function MapView({ data, speciesFilter, speciesColors, imageUrl, isActive, isAnnotated }) {
  const containerRef = useRef(null);
  const imgRef = useRef(null);
  const [imgNatural, setImgNatural] = useState({ w: 0, h: 0 });
  const [renderArea, setRenderArea] = useState({ x: 0, y: 0, w: 0, h: 0 });
  const [hoveredFeat, setHoveredFeat] = useState(null);
  const [tooltipPos, setTooltipPos] = useState({ x: 0, y: 0 });
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const [showLabels, setShowLabels] = useState(true);
  const [imgLoaded, setImgLoaded] = useState(false);

  // Compute the actual rendered area of the image inside the container
  // (accounts for object-fit: contain letterboxing)
  const computeRenderArea = useCallback(() => {
    const img = imgRef.current;
    const container = containerRef.current;
    if (!img || !container || !imgNatural.w || !imgNatural.h) return;

    const cW = container.clientWidth;
    const cH = container.clientHeight;
    const natW = imgNatural.w;
    const natH = imgNatural.h;

    const imgRatio = natW / natH;
    const conRatio = cW / cH;

    let renderW, renderH, offsetX, offsetY;

    if (imgRatio > conRatio) {
      // Image wider than container → letterbox top/bottom
      renderW = cW;
      renderH = cW / imgRatio;
      offsetX = 0;
      offsetY = (cH - renderH) / 2;
    } else {
      // Image taller than container → pillarbox left/right
      renderH = cH;
      renderW = cH * imgRatio;
      offsetX = (cW - renderW) / 2;
      offsetY = 0;
    }

    setRenderArea({ x: offsetX, y: offsetY, w: renderW, h: renderH });
  }, [imgNatural]);

  // Recompute on resize
  useEffect(() => {
    const obs = new ResizeObserver(computeRenderArea);
    if (containerRef.current) obs.observe(containerRef.current);
    return () => obs.disconnect();
  }, [computeRenderArea]);

  useEffect(() => {
    computeRenderArea();
  }, [imgNatural, computeRenderArea]);

  // Reset zoom/pan when a new image loads
  useEffect(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setImgLoaded(false);
  }, [imageUrl]);

  // Convert pixel coords [px, py] → SVG coords
  const toSVG = useCallback((px, py) => {
    if (!imgNatural.w || !imgNatural.h || !renderArea.w) return [0, 0];
    const scaleX = renderArea.w / imgNatural.w;
    const scaleY = renderArea.h / imgNatural.h;
    return [
      renderArea.x + px * scaleX,
      renderArea.y + py * scaleY,
    ];
  }, [imgNatural, renderArea]);

  // Mouse handlers for pan
  const handleMouseDown = (e) => {
    if (e.button !== 1 && !e.ctrlKey) return; // middle-click or ctrl+drag to pan
    setIsPanning(true);
    setPanStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    e.preventDefault();
  };

  const handleMouseMove = (e) => {
    if (isPanning) {
      setPan({ x: e.clientX - panStart.x, y: e.clientY - panStart.y });
    }
  };

  const handleMouseUp = () => setIsPanning(false);

  const handleWheel = (e) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.85 : 1.18;
    setZoom(z => Math.min(8, Math.max(0.5, z * delta)));
  };

  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  // Filter visible features
  const visibleFeatures = data?.features?.filter(
    f => speciesFilter[f.species] !== false
  ) ?? [];

  const hasImage = !!imageUrl;
  const hasDetections = visibleFeatures.length > 0;

  // No image yet — empty state
  if (!hasImage) {
    return (
      <div className="detection-empty">
        <div className="detection-empty-icon">🛰️</div>
        <div className="detection-empty-title">No Image Loaded</div>
        <div className="detection-empty-sub">
          Select an image from the gallery or upload a file, then click <strong>Run Forest Scan</strong> to see detections here.
        </div>
      </div>
    );
  }

  const containerStyle = {
    width: '100%',
    height: '100%',
    overflow: 'hidden',
    position: 'relative',
    cursor: isPanning ? 'grabbing' : zoom > 1 ? 'grab' : 'default',
    background: '#060a07',
  };

  const innerStyle = {
    width: '100%',
    height: '100%',
    transform: `scale(${zoom}) translate(${pan.x / zoom}px, ${pan.y / zoom}px)`,
    transformOrigin: 'center center',
    transition: isPanning ? 'none' : 'transform 0.15s ease',
    position: 'relative',
  };

  return (
    <div className="detection-wrap">
      {/* Toolbar */}
      <div className="detection-toolbar">
        <div className="detection-toolbar-left">
          {hasDetections && (
            <span className="detection-count-badge">
              {visibleFeatures.length} crown{visibleFeatures.length !== 1 ? 's' : ''} detected
            </span>
          )}
        </div>
        <div className="detection-toolbar-right">
          <button
            className={`dtool-btn ${showLabels ? 'active' : ''}`}
            onClick={() => setShowLabels(l => !l)}
            title="Toggle species labels"
          >
            {showLabels ? <Eye size={15} /> : <EyeOff size={15} />}
            <span>{showLabels ? 'Labels On' : 'Labels Off'}</span>
          </button>
          <button className="dtool-btn" onClick={() => setZoom(z => Math.min(8, z * 1.25))} title="Zoom in">
            <ZoomIn size={15} />
          </button>
          <button className="dtool-btn" onClick={() => setZoom(z => Math.max(0.5, z * 0.8))} title="Zoom out">
            <ZoomOut size={15} />
          </button>
          <button className="dtool-btn" onClick={resetView} title="Reset zoom">
            <Maximize2 size={15} />
          </button>
        </div>
      </div>

      {/* Image + SVG Overlay container */}
      <div
        ref={containerRef}
        style={containerStyle}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
      >
        <div style={innerStyle}>
          {/* Base image */}
          <img
            ref={imgRef}
            src={imageUrl}
            alt="Orthomosaic"
            className="detection-img"
            onLoad={(e) => {
              setImgNatural({ w: e.target.naturalWidth, h: e.target.naturalHeight });
              setImgLoaded(true);
            }}
            draggable={false}
          />

          {/* SVG detection overlay — only shown when NOT using YOLO-annotated image.
              When isAnnotated=true, YOLO has already drawn the boxes natively. */}
          {imgLoaded && renderArea.w > 0 && !isAnnotated && (
            <svg
              className="detection-svg"
              style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                pointerEvents: 'none',
              }}
            >
              {visibleFeatures.map((feat) => {
                const isOther   = feat.species === 'Other';
                const color     = isOther
                  ? '#94A3B8'
                  : (speciesColors[feat.species] || feat.color || '#94A3B8');
                const isHovered = hoveredFeat?.id === feat.id;
                const isReview  = feat.status === 'REVIEW_REQUIRED';

                // Build polygon points string from crown_polygon_px
                const coords = feat.crown_polygon_px || feat.crown_polygon || [];
                const points = coords
                  .map(([px, py]) => toSVG(px, py).join(','))
                  .join(' ');

                // Label position: top-left of bbox
                const [lx, ly] = toSVG(feat.bbox[0], feat.bbox[1]);
                const [rx]     = toSVG(feat.bbox[2], feat.bbox[1]);
                const labelW   = Math.max(90, rx - lx);
                const labelText = isOther
                  ? `Tree (uncertain) · ${(feat.confidence * 100).toFixed(0)}%`
                  : `${feat.species} · ${(feat.confidence * 100).toFixed(0)}%`;
                const fontSize = Math.max(9, Math.min(13, renderArea.w / 60));

                return (
                  <g
                    key={feat.id}
                    style={{ pointerEvents: 'all', cursor: 'pointer' }}
                    onMouseEnter={(e) => {
                      setHoveredFeat(feat);
                      setTooltipPos({ x: e.clientX, y: e.clientY });
                    }}
                    onMouseMove={(e) => {
                      setTooltipPos({ x: e.clientX, y: e.clientY });
                    }}
                    onMouseLeave={() => setHoveredFeat(null)}
                  >
                    {/* Crown polygon fill */}
                    {points && (
                      <polygon
                        points={points}
                        fill={color}
                        fillOpacity={isHovered ? 0.30 : (isOther ? 0.08 : 0.18)}
                        stroke={color}
                        strokeWidth={isHovered ? 2.5 : (isOther ? 1.5 : 1.8)}
                        strokeOpacity={isOther ? 0.65 : 0.92}
                        strokeDasharray={isOther || isReview ? '5 3' : 'none'}
                      />
                    )}

                    {/* Label tag */}
                    {showLabels && (
                      <>
                        <rect
                          x={lx}
                          y={ly - 20}
                          width={labelW}
                          height={20}
                          fill={isOther ? 'rgba(30,35,32,0.82)' : color}
                          fillOpacity={isOther ? 1 : 0.85}
                          stroke={isOther ? '#94A3B8' : 'none'}
                          strokeWidth={isOther ? 1 : 0}
                          rx={3}
                        />
                        <text
                          x={lx + 5}
                          y={ly - 6}
                          fill={isOther ? '#94A3B8' : '#fff'}
                          fontSize={fontSize}
                          fontFamily="'Sora', sans-serif"
                          fontWeight="600"
                        >
                          {labelText}
                        </text>
                      </>
                    )}
                  </g>
                );
              })}
            </svg>
          )}

          {/* Loading shimmer */}
          {!imgLoaded && (
            <div className="detection-loading">
              <div className="gallery-spinner"></div>
              <span>Loading image…</span>
            </div>
          )}
        </div>
      </div>

      {/* Hover tooltip */}
      {hoveredFeat && (
        <div
          className="detection-tooltip"
          style={{
            left: tooltipPos.x + 14,
            top: tooltipPos.y - 10,
          }}
        >
          <div
            className="dt-species"
            style={{ color: speciesColors[hoveredFeat.species] || '#94A3B8' }}
          >
            {hoveredFeat.species}
          </div>
          <div className="dt-row">
            <span>Confidence</span>
            <span className="dt-val">{(hoveredFeat.confidence * 100).toFixed(1)}%</span>
          </div>
          <div className="dt-row">
            <span>Crown Area</span>
            <span className="dt-val">{hoveredFeat.crown_area_px?.toLocaleString()} px²</span>
          </div>
          <div className="dt-row">
            <span>Circularity</span>
            <span className="dt-val">{hoveredFeat.circularity?.toFixed(3)}</span>
          </div>
          <div className="dt-row">
            <span>Status</span>
            <span className={`badge ${hoveredFeat.status === 'AUTO_ACCEPTED' ? 'auto' : hoveredFeat.status === 'MANUALLY_VERIFIED' ? 'manual' : 'review'}`}>
              {hoveredFeat.status === 'AUTO_ACCEPTED' ? 'Auto' : hoveredFeat.status === 'MANUALLY_VERIFIED' ? 'Verified' : 'Review'}
            </span>
          </div>
        </div>
      )}

      {/* Species legend */}
      {hasDetections && (() => {
        const counts = data?.summary?.species_counts || {};
        const items = Object.entries(counts).sort((a, b) => b[1] - a[1]);
        return (
          <div className="map-legend">
            <div className="legend-title">Species</div>
            <div className="legend-items">
              {items.map(([sp, count]) => (
                <div key={sp} className="legend-item">
                  <div className="legend-dot" style={{ backgroundColor: speciesColors[sp] || '#94A3B8' }} />
                  <span>{sp} <span style={{ color: 'var(--text-muted)' }}>({count})</span></span>
                </div>
              ))}
            </div>
          </div>
        );
      })()}
    </div>
  );
}
