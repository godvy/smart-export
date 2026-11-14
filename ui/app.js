/**
 * Smart Export - PyWebview Frontend Client & Bridge Handler
 */
(function () {
  'use strict';

  // State
  const state = {
    selectedFolder: '',
    selectedMode: 1, // Default: By Instrument (1)
    autoRename: true,
    isExporting: false,
  };

  // DOM Elements
  const dropZone = document.getElementById('drop-zone');
  const btnBrowse = document.getElementById('btn-browse');
  const folderInput = document.getElementById('folder-input');
  const folderDisplay = document.getElementById('folder-display');
  const folderName = document.getElementById('folder-name');

  const modeSelector = document.getElementById('mode-selector');
  const modePills = modeSelector ? modeSelector.querySelectorAll('.mode-pill') : [];
  const autoRenameCheckbox = document.getElementById('auto-rename');

  const btnExport = document.getElementById('btn-export');
  const progressSection = document.getElementById('progress-section');
  const progressBar = document.getElementById('progress-bar');
  const progressBarFill = document.getElementById('progress-bar-fill');
  const statusText = document.getElementById('status-text');
  const progressPercent = document.getElementById('progress-percent');

  const successCard = document.getElementById('success-card');
  const successCount = document.getElementById('success-count');
  const stemSummaryList = document.getElementById('stem-summary-list');
  const btnOpenFolder = document.getElementById('btn-open-folder');

  // Helper: check if pywebview api is available
  function getPywebviewApi() {
    if (window.pywebview && window.pywebview.api) {
      return window.pywebview.api;
    }
    return null;
  }

  // Update folder UI
  function setSelectedFolder(path) {
    if (!path) return;
    state.selectedFolder = path;
    if (folderName) {
      folderName.textContent = path;
    }
    if (folderDisplay) {
      folderDisplay.classList.remove('hidden');
    }
    if (dropZone) {
      dropZone.classList.add('has-folder');
    }
  }

  // Select folder handler
  async function handleSelectFolder() {
    const api = getPywebviewApi();
    if (api && typeof api.select_folder === 'function') {
      try {
        const folder = await api.select_folder();
        if (folder) {
          setSelectedFolder(folder);
        }
      } catch (err) {
        console.error('Error selecting folder via pywebview:', err);
      }
    } else if (folderInput) {
      folderInput.click();
    }
  }

  // Drag and drop event listeners
  if (dropZone) {
    ['dragenter', 'dragover'].forEach(function (eventName) {
      dropZone.addEventListener(eventName, function (e) {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add('dragover');
      });
    });

    ['dragleave', 'dragend'].forEach(function (eventName) {
      dropZone.addEventListener(eventName, function (e) {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.remove('dragover');
      });
    });

    dropZone.addEventListener('drop', function (e) {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('dragover');

      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        const firstFile = e.dataTransfer.files[0];
        // In desktop pywebview / electron environments, file.path is available
        const folderPath = firstFile.path
          ? firstFile.path.replace(/[\\/][^\\/]+$/, '')
          : (firstFile.webkitRelativePath ? firstFile.webkitRelativePath.split('/')[0] : firstFile.name);
        setSelectedFolder(folderPath || firstFile.name);
      }
    });

    dropZone.addEventListener('click', function (e) {
      // Don't trigger if browse button itself was clicked (it has its own listener)
      if (e.target !== btnBrowse) {
        handleSelectFolder();
      }
    });
  }

  if (btnBrowse) {
    btnBrowse.addEventListener('click', function (e) {
      e.stopPropagation();
      handleSelectFolder();
    });
  }

  if (folderInput) {
    folderInput.addEventListener('change', function (e) {
      if (e.target.files && e.target.files.length > 0) {
        const firstFile = e.target.files[0];
        const relativePath = firstFile.webkitRelativePath || '';
        const folderNameStr = relativePath.split('/')[0] || firstFile.name;
        setSelectedFolder(folderNameStr);
      }
    });
  }

  // Mode Selector Pills
  if (modeSelector) {
    modePills.forEach(function (pill) {
      pill.addEventListener('click', function () {
        modePills.forEach(function (p) {
          p.classList.remove('active');
        });
        pill.classList.add('active');
        const modeAttr = pill.getAttribute('data-mode');
        state.selectedMode = modeAttr !== null ? parseInt(modeAttr, 10) : 1;
      });
    });
  }

  // Auto-rename checkbox
  if (autoRenameCheckbox) {
    autoRenameCheckbox.addEventListener('change', function () {
      state.autoRename = autoRenameCheckbox.checked;
    });
  }

  // Export action
  async function handleStartExport() {
    if (!state.selectedFolder) {
      if (statusText) {
        statusText.textContent = 'Please select an audio folder first.';
      }
      if (progressSection) {
        progressSection.classList.remove('hidden');
      }
      return;
    }

    state.isExporting = true;
    if (btnExport) {
      btnExport.disabled = true;
    }
    if (successCard) {
      successCard.classList.add('hidden');
    }
    if (progressSection) {
      progressSection.classList.remove('hidden');
    }

    window.updateProgress(0, 'Initializing Smart Export engine...');

    const config = {
      folder: state.selectedFolder,
      mode: state.selectedMode,
      auto_rename: autoRenameCheckbox ? autoRenameCheckbox.checked : state.autoRename,
    };

    const api = getPywebviewApi();
    if (api && typeof api.start_export === 'function') {
      try {
        await api.start_export(config);
      } catch (err) {
        console.error('Error starting export:', err);
        window.updateProgress(0, 'Export failed: ' + (err.message || err));
        if (btnExport) {
          btnExport.disabled = false;
        }
      }
    } else {
      console.log('Pywebview API not connected. Mocking export with config:', config);
    }
  }

  if (btnExport) {
    btnExport.addEventListener('click', handleStartExport);
  }

  // Open output folder
  async function handleOpenOutputFolder() {
    const api = getPywebviewApi();
    if (api && typeof api.open_output_folder === 'function') {
      try {
        await api.open_output_folder();
      } catch (err) {
        console.error('Error opening output folder:', err);
      }
    } else {
      console.log('Opening output folder (mock)');
    }
  }

  if (btnOpenFolder) {
    btnOpenFolder.addEventListener('click', handleOpenOutputFolder);
  }

  // Global Callbacks for Python PyWebview Bridge
  window.updateProgress = function (percent, message) {
    const clampedPercent = Math.min(100, Math.max(0, Math.round(percent || 0)));

    if (progressPercent) {
      progressPercent.textContent = clampedPercent + '%';
    }
    if (progressBarFill) {
      progressBarFill.style.width = clampedPercent + '%';
    }
    if (progressBar) {
      progressBar.setAttribute('aria-valuenow', String(clampedPercent));
    }
    if (statusText && message) {
      statusText.textContent = message;
    }
    if (progressSection) {
      progressSection.classList.remove('hidden');
    }
  };

  window.onExportComplete = function (manifest) {
    state.isExporting = false;
    if (btnExport) {
      btnExport.disabled = false;
    }
    if (progressSection) {
      progressSection.classList.add('hidden');
    }
    if (successCard) {
      successCard.classList.remove('hidden');
    }

    const data = manifest || {};
    const totalStems =
      data.total_output_stems !== undefined
        ? data.total_output_stems
        : Array.isArray(data.stems)
        ? data.stems.length
        : 0;

    if (successCount) {
      successCount.textContent = totalStems + (totalStems === 1 ? ' stem exported successfully' : ' stems exported successfully');
    }

    if (stemSummaryList) {
      stemSummaryList.innerHTML = '';
      const stems = Array.isArray(data.stems) ? data.stems : [];

      if (stems.length === 0) {
        const li = document.createElement('li');
        li.className = 'summary-item empty';
        li.textContent = 'Export completed without individual stems recorded.';
        stemSummaryList.appendChild(li);
      } else {
        stems.forEach(function (stem) {
          const li = document.createElement('li');
          li.className = 'summary-item';

          const nameSpan = document.createElement('span');
          nameSpan.className = 'stem-name';
          nameSpan.textContent = stem.stem_file || 'Stem.wav';

          const metaSpan = document.createElement('span');
          metaSpan.className = 'stem-meta';
          const trackCount = stem.track_count || (stem.source_tracks ? stem.source_tracks.length : 1);
          metaSpan.textContent = trackCount + (trackCount === 1 ? ' track' : ' tracks');

          li.appendChild(nameSpan);
          li.appendChild(metaSpan);
          stemSummaryList.appendChild(li);
        });
      }
    }
  };
})();
