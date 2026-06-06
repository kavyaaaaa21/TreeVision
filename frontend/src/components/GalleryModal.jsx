import React, { useState, useEffect } from 'react';
import { Search, X, Image as ImageIcon, FileText } from 'lucide-react';

export default function GalleryModal({ isOpen, onClose, onSelect, selectedName }) {
  const [images, setImages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [activeFilter, setActiveFilter] = useState('all'); // 'all', 'tiff', 'other'
  const [dirPath, setDirPath] = useState('data/images/');

  useEffect(() => {
    if (isOpen) {
      loadImages();
    }
  }, [isOpen]);

  const loadImages = async () => {
    setLoading(true);
    try {
      const res = await fetch('/api/images');
      if (!res.ok) throw new Error('Failed to fetch image list');
      const data = await res.json();
      setImages(data.images || []);
      if (data.directory) {
        setDirPath(data.directory);
      }
    } catch (err) {
      console.error('[gallery] Load failed:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleCardClick = (name) => {
    onSelect(name);
  };

  const handleCardDblClick = (name) => {
    onSelect(name);
    onClose();
  };

  // Filter logic
  const filteredImages = images.filter((img) => {
    // 1. Extension filter
    if (activeFilter === 'tiff' && !img.is_tiff) return false;
    if (activeFilter === 'other' && img.is_tiff) return false;
    
    // 2. Search query filter
    if (searchQuery) {
      return img.name.toLowerCase().includes(searchQuery.trim().toLowerCase());
    }
    return true;
  });

  if (!isOpen) return null;

  return (
    <div className={`gallery-overlay ${isOpen ? 'open' : ''}`} onClick={(e) => e.target.classList.contains('gallery-overlay') && onClose()}>
      <div className="gallery-modal">
        
        {/* Header */}
        <div className="gallery-header">
          <div className="gallery-title">
            <ImageIcon size={20} className="text-accent-green" />
            <span>Local Image Library</span>
            <span className="gallery-dir" title="Server directory path">{dirPath}</span>
          </div>
          <div className="gallery-header-right">
            <div style={{ position: 'relative', display: 'flex', alignItems: 'center' }}>
              <Search size={16} style={{ position: 'absolute', left: '12px', color: 'var(--text-muted)' }} />
              <input
                className="gallery-search"
                style={{ paddingLeft: '34px' }}
                placeholder="Search tiles..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            <button className="gallery-close" onClick={onClose} aria-label="Close modal">
              <X size={20} />
            </button>
          </div>
        </div>

        {/* Filters */}
        <div className="gallery-filters">
          <button
            className={`gf-btn ${activeFilter === 'all' ? 'active' : ''}`}
            onClick={() => setActiveFilter('all')}
          >
            All Files
          </button>
          <button
            className={`gf-btn ${activeFilter === 'tiff' ? 'active' : ''}`}
            onClick={() => setActiveFilter('tiff')}
          >
            GeoTIFF Only
          </button>
          <button
            className={`gf-btn ${activeFilter === 'other' ? 'active' : ''}`}
            onClick={() => setActiveFilter('other')}
          >
            JPEG / PNG
          </button>
        </div>

        {/* Grid Area */}
        <div className="gallery-grid">
          {loading ? (
            <div className="gallery-loading">
              <div className="gallery-spinner"></div>
              <span>Scanning data/images on server...</span>
            </div>
          ) : filteredImages.length === 0 ? (
            <div className="gallery-empty">
              <span>No matching images found in <code>data/images/</code>.</span>
              <br />
              <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                Drop image files there and refresh the scan list.
              </span>
            </div>
          ) : (
            filteredImages.map((img) => {
              const isSelected = selectedName === img.name;
              return (
                <div
                  key={img.name}
                  className={`img-card ${isSelected ? 'selected-card' : ''}`}
                  onClick={() => handleCardClick(img.name)}
                  onDoubleClick={() => handleCardDblClick(img.name)}
                >
                  <div className="img-thumb-wrap">
                    <img
                      className="img-thumb loaded"
                      src={`/api/images/${encodeURIComponent(img.name)}/thumb`}
                      alt={img.name}
                      onError={(e) => {
                        e.target.onerror = null;
                        e.target.style.display = 'none';
                        e.target.nextSibling.style.display = 'flex';
                      }}
                    />
                    <div className="img-thumb-placeholder" style={{ display: 'none' }}>
                      <FileText size={24} />
                    </div>
                  </div>
                  <div className="img-card-body">
                    <div className="img-card-name" title={img.name}>
                      {img.name}
                    </div>
                    <div className="img-card-meta">
                      <span className={`img-card-ext ${img.is_tiff ? 'tiff' : ''}`}>
                        {img.ext.replace('.', '')}
                      </span>
                      <span className="img-card-size">{img.size_mb} MB</span>
                    </div>
                    <button className="img-card-select">
                      {isSelected ? 'Selected' : 'Select'}
                    </button>
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Footer */}
        <div className="gallery-footer">
          <span>{filteredImages.length} of {images.length} files found</span>
          <span>Images are read directly from <code>data/images/</code> on your host</span>
        </div>

      </div>
    </div>
  );
}
