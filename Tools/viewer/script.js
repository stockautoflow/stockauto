// Day.jsが読み込まれているか確認 (HTML側で読み込み前提)
if (typeof dayjs === 'undefined' || typeof dayjs.extend === 'undefined' || typeof dayjs_plugin_customParseFormat === 'undefined') {
    console.error("Day.js or its 'customParseFormat' plugin is not loaded. Please include it in your HTML.");
    alert("Day.jsライブラリが読み込まれていません。日付関連機能が正しく動作しない可能性があります。");
} else {
    dayjs.extend(dayjs_plugin_customParseFormat);
}

document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const svgFilesInput = document.getElementById('svgFilesInput');
    const svgContainer = document.getElementById('svgContainer');
    const fileListElement = document.getElementById('fileList');
    const crosshairXEl = document.getElementById('crosshairX');
    const crosshairYEl = document.getElementById('crosshairY');
    const crosshairInfoEl = document.getElementById('crosshairInfo');
    const placeholderText = svgContainer.querySelector('.placeholder-text');

    // Action Buttons
    const undoBtn = document.getElementById('undoBtn');
    const redoBtn = document.getElementById('redoBtn');
    const saveAnnotationsBtn = document.getElementById('saveAnnotationsBtn');
    const loadAnnotationsInput = document.getElementById('loadAnnotationsInput');


    // Data Range Inputs
    const dataXMinInput = document.getElementById('dataXMin');
    const dataXMaxInput = document.getElementById('dataXMax');
    const dataYMinInput = document.getElementById('dataYMin');
    const dataYMaxInput = document.getElementById('dataYMax');
    const applyDataRangeBtn = document.getElementById('applyDataRange');
    const clearSettingsButton = document.getElementById('clearSettingsButton');

    // Drawing Tools
    const toolSelectModeBtn = document.getElementById('toolSelectMode');
    const toolTrendlineBtn = document.getElementById('toolTrendline');
    const toolHorizontalLineBtn = document.getElementById('toolHorizontalLine');
    const toolVerticalLineBtn = document.getElementById('toolVerticalLine');
    const toolThickDotBtn = document.getElementById('toolThickDot');
    const deleteSelectedObjectBtn = document.getElementById('deleteSelectedObject');
    const drawingColorInput = document.getElementById('drawingColor');
    const drawingStrokeWidthInput = document.getElementById('drawingStrokeWidth');

    // Selected Object Properties
    const selectedObjectPanel = document.getElementById('selectedObjectPanel');
    const selectedObjectColorInput = document.getElementById('selectedObjectColor');
    const selectedObjectStrokeWidthInput = document.getElementById('selectedObjectStrokeWidth');


    // --- State Variables ---
    let svgElement = null;
    let drawingLayer = null;

    let currentScale = 1;
    let panX = 0;
    let panY = 0;

    let isDragging = false; // For panning
    let dragStartX, dragStartY;
    let initialPanX, initialPanY;

    let storedFiles = [];
    let currentFileIndex = -1;
    let currentFileName = '';

    let dataXMin = 0, dataXMax = 100, dataYMin = 0, dataYMax = 100;
    let dataXMinIsDate = true;
    const DATE_FORMAT = 'YYYY-MM-DD HH:mm';
    const INTERNAL_DATE_AS_NUMBER = true;

    let svgViewBox = { x: 0, y: 0, width: 100, height: 100 };
    let svgViewBoxWidth = 100, svgViewBoxHeight = 100;

    let drawingMode = 'select';
    let drawnObjects = [];
    let tempDrawingElement = null;
    let selectedObjectIndex = -1;
    let draggedObject = null;
    let objectIdCounter = 0;

    let isDraggingObject = false;
    let dragObjectStartX_svg, dragObjectStartY_svg;
    let draggedObjectInitialPos = {};

    // Undo/Redo
    let historyStack = [];
    let historyPointer = -1;
    const MAX_HISTORY_STATES = 50;


    // --- Initialization ---
    loadInitialSettings();
    updateToolButtonsState();
    updateUndoRedoButtons();

    // --- Undo/Redo Logic ---
    function pushStateToHistory() {
        const currentState = {
            drawnObjects: JSON.parse(JSON.stringify(drawnObjects)),
            objectIdCounter: objectIdCounter
        };
        historyStack = historyStack.slice(0, historyPointer + 1);
        historyStack.push(currentState);
        if (historyStack.length > MAX_HISTORY_STATES) {
            historyStack.shift();
        }
        historyPointer = historyStack.length - 1;
        updateUndoRedoButtons();
    }

    function undo() {
        if (historyPointer <= 0) {
            updateUndoRedoButtons(); return;
        }
        historyPointer--;
        loadStateFromHistory(historyStack[historyPointer]);
        updateUndoRedoButtons();
    }

    function redo() {
        if (historyPointer >= historyStack.length - 1) {
            updateUndoRedoButtons(); return;
        }
        historyPointer++;
        loadStateFromHistory(historyStack[historyPointer]);
        updateUndoRedoButtons();
    }

    function loadStateFromHistory(stateToRestore) {
        if (!stateToRestore) return;
        drawnObjects = JSON.parse(JSON.stringify(stateToRestore.drawnObjects));
        objectIdCounter = stateToRestore.objectIdCounter;
        renderDrawnObjects();
        setSelectedObject(-1);
    }

    function updateUndoRedoButtons() {
        undoBtn.disabled = historyPointer <= 0;
        redoBtn.disabled = historyPointer >= historyStack.length - 1;
    }

    function clearHistory() {
        historyStack = [];
        historyPointer = -1;
        updateUndoRedoButtons();
    }

    // --- File Handling ---
    svgFilesInput.addEventListener('change', handleFileSelection);
    loadAnnotationsInput.addEventListener('change', handleAnnotationFileLoad);


    function handleFileSelection(event) {
        if (currentFileName) saveSettingsForCurrentFile();

        storedFiles = Array.from(event.target.files).filter(file => file.name.endsWith('.svg'));
        if (storedFiles.length === 0 && event.target.files.length > 0) {
            alert('SVGファイルのみ選択可能です。');
        }
        renderFileList();

        if (storedFiles.length > 0) {
            let loadIndex = 0;
            if (currentFileIndex !== -1 && currentFileIndex < storedFiles.length) {
                const oldFileName = currentFileName;
                const newIndexForOldFile = storedFiles.findIndex(f => f.name === oldFileName);
                if (newIndexForOldFile !== -1) {
                    loadIndex = newIndexForOldFile;
                }
            }
            loadSvgFileByIndex(loadIndex);
        } else {
            clearSvgContainerAndState(); // This now calls clearHistory
            currentFileIndex = -1;
            currentFileName = '';
            // clearHistory(); // Called by clearSvgContainerAndState
            updateUndoRedoButtons();
        }
    }

    function renderFileList() {
        fileListElement.innerHTML = '';
        storedFiles.forEach((file, index) => {
            const listItem = document.createElement('li');
            listItem.textContent = file.name;
            listItem.dataset.index = index;
            listItem.addEventListener('click', () => {
                if (currentFileIndex !== index) {
                    if (currentFileName) saveSettingsForCurrentFile();
                    loadSvgFileByIndex(index);
                }
            });
            if (index === currentFileIndex) {
                listItem.classList.add('active');
            }
            fileListElement.appendChild(listItem);
        });
    }

    function loadSvgFileByIndex(index) {
        if (index < 0 || index >= storedFiles.length) return;

        currentFileIndex = index;
        const file = storedFiles[index];
        currentFileName = file.name;
        const reader = new FileReader();

        reader.onload = (e) => {
            if (placeholderText) placeholderText.style.display = 'none';
            svgContainer.innerHTML = e.target.result;
            svgElement = svgContainer.querySelector('svg');

            if (svgElement) {
                if (drawingLayer && drawingLayer.parentNode) {
                    drawingLayer.parentNode.removeChild(drawingLayer);
                }
                drawingLayer = document.createElementNS("http://www.w3.org/2000/svg", "svg");
                drawingLayer.classList.add('drawing-layer');
                drawingLayer.setAttribute('width', '100%');
                drawingLayer.setAttribute('height', '100%');
                drawingLayer.style.position = 'absolute';
                drawingLayer.style.top = '0';
                drawingLayer.style.left = '0';
                drawingLayer.style.pointerEvents = 'none';
                svgContainer.appendChild(drawingLayer);

                const viewBoxAttr = svgElement.getAttribute('viewBox');
                if (viewBoxAttr) {
                    const parts = viewBoxAttr.split(/[ ,]+/).map(parseFloat);
                    if (parts.length === 4 && parts[2] > 0 && parts[3] > 0) {
                        svgViewBox = { x: parts[0], y: parts[1], width: parts[2], height: parts[3] };
                    } else {
                        console.warn("Malformed viewBox or non-positive width/height. Using fallback.");
                        const w = parseFloat(svgElement.getAttribute('width')) || 800;
                        const h = parseFloat(svgElement.getAttribute('height')) || 600;
                        svgViewBox = { x: 0, y: 0, width: w, height: h };
                    }
                } else {
                    const w = parseFloat(svgElement.getAttribute('width'));
                    const h = parseFloat(svgElement.getAttribute('height'));
                    if (w > 0 && h > 0) {
                        svgViewBox = { x: 0, y: 0, width: w, height: h };
                        svgElement.setAttribute('viewBox', `0 0 ${w} ${h}`);
                    } else {
                        console.warn("SVG has no viewBox or valid explicit width/height. Using default dimensions.");
                        svgViewBox = { x: 0, y: 0, width: 800, height: 600 };
                        svgElement.setAttribute('viewBox', `0 0 ${svgViewBox.width} ${svgViewBox.height}`);
                    }
                }
                svgViewBoxWidth = svgViewBox.width;
                svgViewBoxHeight = svgViewBox.height;

                svgElement.style.transformOrigin = '0 0';
                drawingLayer.style.transformOrigin = '0 0';

                clearHistory();
                loadSettingsForCurrentFile();
                pushStateToHistory(); // Push initial loaded state
                applyTransform();
            } else {
                console.error("Could not find <svg> element in the loaded file.");
                clearSvgContainerAndState(); // Will also clear history
            }
            renderFileList();
            updateUndoRedoButtons();
        };
        reader.onerror = (e) => {
            console.error("Error reading file:", file.name, e);
            alert(`ファイル ${file.name} の読み込みに失敗しました。`);
            clearSvgContainerAndState(); // Will also clear history
        };
        reader.readAsText(file);
    }

    function clearSvgContainerAndState() {
        svgContainer.innerHTML = '';
        if (placeholderText) {
            svgContainer.appendChild(placeholderText);
            placeholderText.style.display = 'block';
        }
        svgElement = null;
        if (drawingLayer && drawingLayer.parentNode) {
            drawingLayer.parentNode.removeChild(drawingLayer);
        }
        drawingLayer = null;
        drawnObjects = [];
        objectIdCounter = 0;
        selectedObjectIndex = -1;
        draggedObject = null;
        resetTransformations();
        updateDataRangeUIWithDefaults(); // Resets UI for data range
        setSelectedObject(-1);
        clearHistory(); // Clear history here
        updateUndoRedoButtons();
    }

    function resetTransformations() {
        currentScale = 1; panX = 0; panY = 0;
    }

    function updateDataRangeUIWithDefaults() {
        const now = dayjs();
        const oneMonthAgo = dayjs().subtract(1, 'month');
        dataXMinInput.value = dataXMinIsDate ? oneMonthAgo.format(DATE_FORMAT) : '0';
        dataXMaxInput.value = dataXMinIsDate ? now.format(DATE_FORMAT) : '100';
        dataYMinInput.value = '0';
        dataYMaxInput.value = '100';
    }

    // --- Coordinate Transformation & Data Range ---
    applyDataRangeBtn.addEventListener('click', () => {
        let xMinRaw = dataXMinInput.value.trim(); let xMaxRaw = dataXMaxInput.value.trim();
        let yMinRaw = dataYMinInput.value.trim(); let yMaxRaw = dataYMaxInput.value.trim();
        let pXMin, pXMax, pYMin, pYMax;

        if (dataXMinIsDate) {
            let tXMin = dayjs(xMinRaw, DATE_FORMAT, true); let tXMax = dayjs(xMaxRaw, DATE_FORMAT, true);
            if (!tXMin.isValid()) { alert(`X軸 最小値の日付形式無効 (${DATE_FORMAT})`); return; }
            if (!tXMax.isValid()) { alert(`X軸 最大値の日付形式無効 (${DATE_FORMAT})`); return; }
            pXMin = tXMin.valueOf(); pXMax = tXMax.valueOf();
        } else {
            pXMin = parseFloat(xMinRaw); pXMax = parseFloat(xMaxRaw);
        }
        pYMin = parseFloat(yMinRaw); pYMax = parseFloat(yMaxRaw);

        if (isNaN(pXMin) || isNaN(pXMax) || isNaN(pYMin) || isNaN(pYMax) || pXMax <= pXMin || pYMax <= pYMin) {
            alert("データ範囲の値が無効です"); return;
        }
        dataXMin = pXMin; dataXMax = pXMax; dataYMin = pYMin; dataYMax = pYMax;
        saveSettingsForCurrentFile();
        // Data range changes are not part of drawing undo/redo for now. If they were, call pushStateToHistory()
    });

    function mapSvgToDataCoords(svgX, svgY) {
        if (!svgElement || svgViewBoxWidth <= 0 || svgViewBoxHeight <= 0 || dataXMax === dataXMin || dataYMax === dataYMin) {
            return { x: null, y: null, xRaw: null };
        }
        const dataXRaw = dataXMin + (svgX / svgViewBoxWidth) * (dataXMax - dataXMin);
        const dataY = dataYMax - (svgY / svgViewBoxHeight) * (dataYMax - dataYMin);
        if (dataXMinIsDate && INTERNAL_DATE_AS_NUMBER) {
            return { x: dayjs(dataXRaw).format(DATE_FORMAT), y: dataY, xRaw: dataXRaw };
        }
        return { x: dataXRaw, y: dataY, xRaw: dataXRaw };
    }

    function mapDataToSvgCoords(dataXValue, dataYValue) {
        let numDataX = dataXValue;
        if (dataXMinIsDate && typeof dataXValue === 'string' && INTERNAL_DATE_AS_NUMBER) {
            const pDate = dayjs(dataXValue, DATE_FORMAT);
            if (!pDate.isValid()) return { x: null, y: null };
            numDataX = pDate.valueOf();
        } else if (dataXMinIsDate && dataXValue instanceof dayjs && INTERNAL_DATE_AS_NUMBER) {
             numDataX = dataXValue.valueOf();
        }
        if (!svgElement || svgViewBoxWidth <= 0 || svgViewBoxHeight <= 0 || dataXMax === dataXMin || dataYMax === dataYMin) {
            return { x: null, y: null };
        }
        const svgX = ((numDataX - dataXMin) / (dataXMax - dataXMin)) * svgViewBoxWidth;
        const svgY = ((dataYMax - dataYValue) / (dataYMax - dataYMin)) * svgViewBoxHeight;
        return { x: svgX, y: svgY };
    }

    function getMousePositionInSvg(event) {
        if (!svgContainer || !svgElement) return null;
        const rect = svgContainer.getBoundingClientRect();
        const cX = event.clientX - rect.left; const cY = event.clientY - rect.top;
        return { x: (cX - panX) / currentScale, y: (cY - panY) / currentScale };
    }

    // --- Pan and Zoom ---
    function applyTransform() {
        if (svgElement) {
            const tVal = `translate(${panX}px, ${panY}px) scale(${currentScale})`;
            svgElement.style.transform = tVal;
            if (drawingLayer) drawingLayer.style.transform = tVal;
        }
    }

    svgContainer.addEventListener('wheel', (event) => {
        if (!svgElement) return;
        event.preventDefault(); event.stopPropagation();
        const rect = svgContainer.getBoundingClientRect();
        const mX_c = event.clientX - rect.left; const mY_c = event.clientY - rect.top;
        const svgMX_b = (mX_c - panX) / currentScale; const svgMY_b = (mY_c - panY) / currentScale;
        const scaleFactor = 1.1;
        if (event.deltaY > 0) { currentScale *= scaleFactor; } else { currentScale /= scaleFactor; }
        currentScale = Math.max(0.02, Math.min(currentScale, 200));
        panX = mX_c - (svgMX_b * currentScale); panY = mY_c - (svgMY_b * currentScale);
        applyTransform(); debouncedSaveSettings();
    }, { passive: false });

    svgContainer.addEventListener('mousedown', (event) => {
        if (!svgElement || event.button !== 0) return;
        const mPosSvg = getMousePositionInSvg(event);
        if (!mPosSvg) return;
        const drawColor = drawingColorInput.value;
        const drawStrokeWidth = parseInt(drawingStrokeWidthInput.value, 10) || 2;

        if (drawingMode === 'select') {
            const clickedInfo = findObjectAtSvgPoint(mPosSvg.x, mPosSvg.y);
            if (clickedInfo) {
                setSelectedObject(clickedInfo.index);
                isDraggingObject = true; draggedObject = drawnObjects[clickedInfo.index];
                dragObjectStartX_svg = mPosSvg.x; dragObjectStartY_svg = mPosSvg.y;
                draggedObjectInitialPos = {
                    x1: draggedObject.x1, y1: draggedObject.y1, x2: draggedObject.x2, y2: draggedObject.y2,
                    y: draggedObject.y, x: draggedObject.x,
                    cx: draggedObject.cx, cy: draggedObject.cy, r: draggedObject.r
                };
                svgContainer.style.cursor = 'move';
            } else {
                setSelectedObject(-1); isDragging = true;
                dragStartX = event.clientX; dragStartY = event.clientY;
                initialPanX = panX; initialPanY = panY;
                svgContainer.style.cursor = 'grabbing';
            }
        } else if (drawingMode === 'trendline_start') {
            tempDrawingElement = document.createElementNS("http://www.w3.org/2000/svg", "line");
            tempDrawingElement.setAttribute('x1', mPosSvg.x); tempDrawingElement.setAttribute('y1', mPosSvg.y);
            tempDrawingElement.setAttribute('x2', mPosSvg.x); tempDrawingElement.setAttribute('y2', mPosSvg.y);
            tempDrawingElement.setAttribute('stroke', drawColor);
            tempDrawingElement.setAttribute('stroke-width', drawStrokeWidth.toString());
            tempDrawingElement.style.pointerEvents = 'none';
            drawingLayer.appendChild(tempDrawingElement);
            drawingMode = 'trendline_draw'; svgContainer.style.cursor = 'crosshair';
        } else if (drawingMode === 'horizontal_line') {
            const id = `obj_${objectIdCounter++}`;
            drawnObjects.push({ id, type: 'horizontal_line', y: mPosSvg.y, color: drawColor, strokeWidth: drawStrokeWidth });
            renderDrawnObjects(); setSelectedObject(drawnObjects.length - 1);
            debouncedSaveSettings(); pushStateToHistory();
        } else if (drawingMode === 'vertical_line') {
            const id = `obj_${objectIdCounter++}`;
            drawnObjects.push({ id, type: 'vertical_line', x: mPosSvg.x, color: drawColor, strokeWidth: drawStrokeWidth });
            renderDrawnObjects(); setSelectedObject(drawnObjects.length - 1);
            debouncedSaveSettings(); pushStateToHistory();
        } else if (drawingMode === 'thick_dot') {
            const id = `obj_${objectIdCounter++}`;
            drawnObjects.push({ id, type: 'thick_dot', cx: mPosSvg.x, cy: mPosSvg.y, r: drawStrokeWidth, color: drawColor });
            renderDrawnObjects(); setSelectedObject(drawnObjects.length - 1);
            debouncedSaveSettings(); pushStateToHistory();
        }
    });

    document.addEventListener('mousemove', (event) => {
        const mPosSvg = getMousePositionInSvg(event);
        const cX = event.clientX; const cY = event.clientY;
        crosshairXEl.style.top = `${cY}px`; crosshairYEl.style.left = `${cX}px`;
        if (svgElement && mPosSvg) {
            const dCoords = mapSvgToDataCoords(mPosSvg.x, mPosSvg.y);
            if (dCoords.x !== null && dCoords.y !== null) {
                const dX = dataXMinIsDate && INTERNAL_DATE_AS_NUMBER ? dayjs(dCoords.xRaw).format(DATE_FORMAT) : (typeof dCoords.x === 'number' ? dCoords.x.toFixed(2) : dCoords.x);
                crosshairInfoEl.textContent = `X: ${dX}, Y: ${dCoords.y.toFixed(2)}`;
            } else {
                crosshairInfoEl.textContent = `X: ${mPosSvg.x.toFixed(0)}, Y: ${mPosSvg.y.toFixed(0)} (SVG Coords)`;
            }
        } else { crosshairInfoEl.textContent = `X: -, Y: -`; }

        if (isDraggingObject && draggedObject && mPosSvg) {
            event.preventDefault();
            const dx_s = mPosSvg.x - dragObjectStartX_svg; const dy_s = mPosSvg.y - dragObjectStartY_svg;
            if (draggedObject.type === 'trendline') {
                draggedObject.x1 = draggedObjectInitialPos.x1 + dx_s; draggedObject.y1 = draggedObjectInitialPos.y1 + dy_s;
                draggedObject.x2 = draggedObjectInitialPos.x2 + dx_s; draggedObject.y2 = draggedObjectInitialPos.y2 + dy_s;
            } else if (draggedObject.type === 'horizontal_line') { draggedObject.y = draggedObjectInitialPos.y + dy_s; }
            else if (draggedObject.type === 'vertical_line') { draggedObject.x = draggedObjectInitialPos.x + dx_s; }
            else if (draggedObject.type === 'thick_dot') {
                draggedObject.cx = draggedObjectInitialPos.cx + dx_s; draggedObject.cy = draggedObjectInitialPos.cy + dy_s;
            }
            renderDrawnObjects();
        } else if (isDragging && drawingMode === 'select' && !isDraggingObject) {
            event.preventDefault();
            const dx_c = cX - dragStartX; const dy_c = cY - dragStartY;
            panX = initialPanX + dx_c; panY = initialPanY + dy_c;
            applyTransform();
        } else if (drawingMode === 'trendline_draw' && tempDrawingElement && mPosSvg) {
            tempDrawingElement.setAttribute('x2', mPosSvg.x); tempDrawingElement.setAttribute('y2', mPosSvg.y);
        }
    });

    document.addEventListener('mouseup', (event) => {
        if (isDraggingObject) {
            isDraggingObject = false;
            svgContainer.style.cursor = (drawingMode === 'select') ? 'grab' : 'crosshair';
            debouncedSaveSettings(); pushStateToHistory();
        } else if (isDragging && drawingMode === 'select' && !isDraggingObject) {
            isDragging = false; svgContainer.style.cursor = 'grab';
            debouncedSaveSettings();
            // Pan does not go to history for now
        } else if (drawingMode === 'trendline_draw' && tempDrawingElement) {
            const x1 = parseFloat(tempDrawingElement.getAttribute('x1')); const y1 = parseFloat(tempDrawingElement.getAttribute('y1'));
            const x2 = parseFloat(tempDrawingElement.getAttribute('x2')); const y2 = parseFloat(tempDrawingElement.getAttribute('y2'));
            const drawColor = drawingColorInput.value; const drawStrokeWidth = parseInt(drawingStrokeWidthInput.value, 10) || 2;
            if (Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2)) > 2) {
                const id = `obj_${objectIdCounter++}`;
                drawnObjects.push({ id, type: 'trendline', x1, y1, x2, y2, color: drawColor, strokeWidth: drawStrokeWidth });
                setSelectedObject(drawnObjects.length -1);
                debouncedSaveSettings(); pushStateToHistory();
            }
            tempDrawingElement.remove(); tempDrawingElement = null;
            renderDrawnObjects(); setDrawingMode('trendline_start');
        }
    });

    // --- Drawing Tools Logic ---
    toolSelectModeBtn.addEventListener('click', () => setDrawingMode('select'));
    toolTrendlineBtn.addEventListener('click', () => setDrawingMode('trendline_start'));
    toolHorizontalLineBtn.addEventListener('click', () => setDrawingMode('horizontal_line'));
    toolVerticalLineBtn.addEventListener('click', () => setDrawingMode('vertical_line'));
    toolThickDotBtn.addEventListener('click', () => setDrawingMode('thick_dot'));

    function setDrawingMode(mode) {
        drawingMode = mode;
        if (tempDrawingElement) { tempDrawingElement.remove(); tempDrawingElement = null; }
        svgContainer.style.cursor = (mode === 'select') ? 'grab' : 'crosshair';
        updateToolButtonsState();
    }

    function updateToolButtonsState() {
        toolSelectModeBtn.classList.toggle('active-tool', drawingMode === 'select');
        toolTrendlineBtn.classList.toggle('active-tool', drawingMode === 'trendline_start' || drawingMode === 'trendline_draw');
        toolHorizontalLineBtn.classList.toggle('active-tool', drawingMode === 'horizontal_line');
        toolVerticalLineBtn.classList.toggle('active-tool', drawingMode === 'vertical_line');
        toolThickDotBtn.classList.toggle('active-tool', drawingMode === 'thick_dot');
    }

    function renderDrawnObjects() {
        if (!drawingLayer) return;
        drawingLayer.innerHTML = '';
        drawnObjects.forEach((obj, index) => {
            let el;
            const color = obj.color || drawingColorInput.value;
            const strokeW = obj.strokeWidth || (obj.type === 'thick_dot' ? 1 : 2); // For lines
            const radius = obj.r || (obj.type === 'thick_dot' ? (parseInt(drawingStrokeWidthInput.value, 10) || 5) : null); // For dots

            if (obj.type === 'trendline') {
                el = document.createElementNS("http://www.w3.org/2000/svg", "line");
                el.setAttribute('x1', obj.x1); el.setAttribute('y1', obj.y1);
                el.setAttribute('x2', obj.x2); el.setAttribute('y2', obj.y2);
                el.setAttribute('stroke', color); el.setAttribute('stroke-width', strokeW.toString());
            } else if (obj.type === 'horizontal_line') {
                el = document.createElementNS("http://www.w3.org/2000/svg", "line");
                el.setAttribute('x1', svgViewBox.x); el.setAttribute('y1', obj.y);
                el.setAttribute('x2', svgViewBox.x + svgViewBox.width); el.setAttribute('y2', obj.y);
                el.setAttribute('stroke', color); el.setAttribute('stroke-width', strokeW.toString());
            } else if (obj.type === 'vertical_line') {
                el = document.createElementNS("http://www.w3.org/2000/svg", "line");
                el.setAttribute('x1', obj.x); el.setAttribute('y1', svgViewBox.y);
                el.setAttribute('x2', obj.x); el.setAttribute('y2', svgViewBox.y + svgViewBox.height);
                el.setAttribute('stroke', color); el.setAttribute('stroke-width', strokeW.toString());
            } else if (obj.type === 'thick_dot') {
                el = document.createElementNS("http://www.w3.org/2000/svg", "circle");
                el.setAttribute('cx', obj.cx); el.setAttribute('cy', obj.cy);
                el.setAttribute('r', radius.toString());
                el.setAttribute('fill', color);
            }
            if (el) {
                el.dataset.objectId = obj.id; el.dataset.objectIndex = index.toString();
                el.style.pointerEvents = 'auto';
                if (index === selectedObjectIndex) el.classList.add('selected-object');
                drawingLayer.appendChild(el);
            }
        });
    }

    function distanceSq(p1x, p1y, p2x, p2y) { return Math.pow(p1x - p2x, 2) + Math.pow(p1y - p2y, 2); }
    function distanceToLineSegmentSq(px, py, x1, y1, x2, y2) {
        const l2 = distanceSq(x1, y1, x2, y2); if (l2 === 0) return distanceSq(px, py, x1, y1);
        let t = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / l2;
        t = Math.max(0, Math.min(1, t));
        return distanceSq(px, py, x1 + t * (x2 - x1), y1 + t * (y2 - y1));
    }

    function findObjectAtSvgPoint(svgX, svgY) {
        const tol = 8 / currentScale; const tolSq = tol * tol;
        let closestInfo = null; let minDistSq = Infinity;
        for (let i = drawnObjects.length - 1; i >= 0; i--) {
            const obj = drawnObjects[i]; let dSq = Infinity;
            if (obj.type === 'trendline') dSq = distanceToLineSegmentSq(svgX, svgY, obj.x1, obj.y1, obj.x2, obj.y2);
            else if (obj.type === 'horizontal_line') { if (svgX >= svgViewBox.x && svgX <= svgViewBox.x + svgViewBox.width) dSq = Math.pow(svgY - obj.y, 2); }
            else if (obj.type === 'vertical_line') { if (svgY >= svgViewBox.y && svgY <= svgViewBox.y + svgViewBox.height) dSq = Math.pow(svgX - obj.x, 2); }
            else if (obj.type === 'thick_dot') {
                const dToCenterSq = distanceSq(svgX, svgY, obj.cx, obj.cy);
                const rEff = (obj.r || 5) + tol; // effective radius for hit
                if (dToCenterSq < rEff * rEff) dSq = dToCenterSq; // use actual distance for sorting if hit
            }
            if (dSq < tolSq && dSq < minDistSq && obj.type !== 'thick_dot') { // Lines
                minDistSq = dSq; closestInfo = { index: i, object: obj };
            } else if (obj.type === 'thick_dot' && dSq !== Infinity && dSq < minDistSq ) { // Dots
                minDistSq = dSq; closestInfo = { index: i, object: obj };
            }
        }
        return closestInfo ? { index: closestInfo.index } : null;
    }

    function setSelectedObject(index) {
        selectedObjectIndex = index;
        deleteSelectedObjectBtn.disabled = (index === -1);
        if (index !== -1 && drawnObjects[index]) {
            const obj = drawnObjects[index];
            selectedObjectPanel.style.display = 'block';
            selectedObjectColorInput.value = obj.color || drawingColorInput.value;
            const labelEl = selectedObjectStrokeWidthInput.previousElementSibling; // Assumes label is previous sibling
            if (obj.type === 'thick_dot') {
                selectedObjectStrokeWidthInput.value = (obj.r || 5).toString();
                if (labelEl) labelEl.textContent = "半径:";
            } else {
                selectedObjectStrokeWidthInput.value = (obj.strokeWidth || 2).toString();
                if (labelEl) labelEl.textContent = "太さ:";
            }
        } else { selectedObjectPanel.style.display = 'none'; }
        renderDrawnObjects();
    }

    deleteSelectedObjectBtn.addEventListener('click', () => {
        if (selectedObjectIndex !== -1 && drawnObjects[selectedObjectIndex]) {
            drawnObjects.splice(selectedObjectIndex, 1);
            setSelectedObject(-1);
            debouncedSaveSettings(); pushStateToHistory();
        }
    });
    selectedObjectColorInput.addEventListener('input', (event) => {
        if (selectedObjectIndex !== -1 && drawnObjects[selectedObjectIndex]) {
            drawnObjects[selectedObjectIndex].color = event.target.value; renderDrawnObjects();
            debouncedSaveSettings(); pushStateToHistory();
        }
    });
    selectedObjectStrokeWidthInput.addEventListener('input', (event) => {
        if (selectedObjectIndex !== -1 && drawnObjects[selectedObjectIndex]) {
            const obj = drawnObjects[selectedObjectIndex]; const newVal = parseInt(event.target.value, 10);
            if (!isNaN(newVal) && newVal > 0) {
                if (obj.type === 'thick_dot') obj.r = newVal; else obj.strokeWidth = newVal;
                renderDrawnObjects(); debouncedSaveSettings(); pushStateToHistory();
            }
        }
    });

    // --- Settings Persistence (localStorage) ---
    const SETTINGS_KEY_PREFIX = 'svgChartViewerSettings_v2.3_';
    let saveTimeout;
    function debouncedSaveSettings() { clearTimeout(saveTimeout); saveTimeout = setTimeout(saveSettingsForCurrentFile, 800); }
    function saveSettingsForCurrentFile() {
        if (!currentFileName) return;
        const settings = {
            panX, panY, currentScale, dataXMin, dataXMax, dataYMin, dataYMax, dataXMinIsDate,
            drawnObjects: drawnObjects.map(o => ({ ...o })), // Simple deep copy for plain objects
            objectIdCounter
        };
        try { localStorage.setItem(SETTINGS_KEY_PREFIX + currentFileName, JSON.stringify(settings)); }
        catch (e) { console.error('LS Save Error:', e); }
    }
    function loadSettingsForCurrentFile() {
        resetApplicationStateForNewFileLoad(); // Start with defaults
        if (!currentFileName) { updateDataRangeUIFromState(); return; }
        try {
            const saved = localStorage.getItem(SETTINGS_KEY_PREFIX + currentFileName);
            if (saved) {
                const s = JSON.parse(saved);
                panX = s.panX || 0; panY = s.panY || 0; currentScale = s.currentScale || 1;
                dataXMinIsDate = typeof s.dataXMinIsDate === 'boolean' ? s.dataXMinIsDate : true;
                if (s.dataXMin !== undefined) {
                    dataXMin = s.dataXMin; dataXMax = s.dataXMax; dataYMin = s.dataYMin; dataYMax = s.dataYMax;
                }
                drawnObjects = s.drawnObjects || [];
                objectIdCounter = s.objectIdCounter || 0;
                if (drawnObjects.length > 0 && objectIdCounter <= drawnObjects.length) {
                     objectIdCounter = Math.max(0, ...drawnObjects.map(o => parseInt((o.id || "obj_0").split('_')[1]))) + 1 || drawnObjects.length + 1;
                }
            }
        } catch (e) { console.error('LS Load Error:', e); }
        updateDataRangeUIFromState();
    }
    function resetApplicationStateForNewFileLoad() {
        resetTransformations();
        const now = dayjs(); const prevMonth = dayjs().subtract(1, 'month');
        dataXMinIsDate = true;
        dataXMin = prevMonth.valueOf(); dataXMax = now.valueOf();
        dataYMin = 0; dataYMax = 100;
        drawnObjects = []; objectIdCounter = 0; selectedObjectIndex = -1; draggedObject = null;
    }
    function updateDataRangeUIFromState() {
        const formatIfNeeded = (val, isDate) => isDate && INTERNAL_DATE_AS_NUMBER && typeof val === 'number' ? dayjs(val).format(DATE_FORMAT) : (val ?? '').toString();
        dataXMinInput.value = formatIfNeeded(dataXMin, dataXMinIsDate);
        dataXMaxInput.value = formatIfNeeded(dataXMax, dataXMinIsDate);
        dataYMinInput.value = (dataYMin ?? '').toString();
        dataYMaxInput.value = (dataYMax ?? '').toString();
    }
    clearSettingsButton.addEventListener('click', () => {
        if (currentFileName && confirm(`'${currentFileName}' の保存設定をクリアしますか？`)) {
            try {
                localStorage.removeItem(SETTINGS_KEY_PREFIX + currentFileName);
                resetApplicationStateForNewFileLoad(); clearHistory();
                updateDataRangeUIFromState(); renderDrawnObjects(); applyTransform();
                pushStateToHistory(); setSelectedObject(-1);
                alert(`'${currentFileName}' 設定クリア完了`);
            } catch (e) { console.error('LS Clear Error:', e); alert('設定クリアエラー'); }
        } else if (!currentFileName) { alert('ファイル未選択'); }
    });

    // --- Save/Load Annotations to/from JSON file ---
    saveAnnotationsBtn.addEventListener('click', () => {
        if (!currentFileName) { alert('SVGファイルを開いてください'); return; }
        if (drawnObjects.length === 0 && !confirm("描画がありません。設定のみ保存しますか？")) return;
        const annotations = {
            version: "SVGViewerAnnotations_v1.0", sourceSvgFile: currentFileName, savedAt: dayjs().toISOString(),
            panX, panY, currentScale, dataXMin, dataXMax, dataYMin, dataYMax, dataXMinIsDate,
            drawnObjects: drawnObjects.map(o => ({ ...o })), objectIdCounter
        };
        try {
            const jsonStr = JSON.stringify(annotations, null, 2);
            const blob = new Blob([jsonStr], { type: 'application/json' });
            const fName = currentFileName.replace(/\.svg$/i, '').replace(/[^a-z0-9_.-]/gi, '_') || 'annotations';
            const dlName = `${fName}.annotations.json`;
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob); link.download = dlName;
            document.body.appendChild(link); link.click(); document.body.removeChild(link);
            URL.revokeObjectURL(link.href);
        } catch (e) { console.error("アノテーション保存エラー:", e); alert("アノテーション保存エラー"); }
    });

    function handleAnnotationFileLoad(event) {
        if (!currentFileName) { alert('まずSVGファイルを開いてください'); event.target.value = ''; return; }
        const file = event.target.files[0]; if (!file) { event.target.value = ''; return; }
        const reader = new FileReader();
        reader.onload = (e_reader) => {
            try {
                const ann = JSON.parse(e_reader.target.result);
                if (!ann || ann.version !== "SVGViewerAnnotations_v1.0" || !ann.drawnObjects) {
                    alert('無効なアノテーションファイル形式'); return;
                }
                if (ann.sourceSvgFile && ann.sourceSvgFile !== currentFileName &&
                    !confirm(`アノテーションは元々 '${ann.sourceSvgFile}' 用です。\n'${currentFileName}' に適用しますか？`)) {
                    return;
                }
                panX = ann.panX || 0; panY = ann.panY || 0; currentScale = ann.currentScale || 1;
                dataXMinIsDate = typeof ann.dataXMinIsDate === 'boolean' ? ann.dataXMinIsDate : true;
                if (ann.dataXMin !== undefined) {
                    dataXMin = ann.dataXMin; dataXMax = ann.dataXMax;
                    dataYMin = ann.dataYMin; dataYMax = ann.dataYMax;
                }
                updateDataRangeUIFromState();
                drawnObjects = ann.drawnObjects || [];
                objectIdCounter = ann.objectIdCounter || 0;
                if (drawnObjects.length > 0) {
                     const maxId = Math.max(0, ...drawnObjects.map(obj => parseInt((obj.id || "obj_0").split('_')[1])));
                     objectIdCounter = Math.max(objectIdCounter, maxId + 1);
                }
                clearHistory(); renderDrawnObjects(); applyTransform();
                pushStateToHistory(); setSelectedObject(-1);
                saveSettingsForCurrentFile(); // Save loaded annotations to localStorage for current SVG
                alert('アノテーション読込完了');
            } catch (err) { console.error("アノテーション処理エラー:", err); alert(`アノテーション処理エラー: ${err.message}`); }
            finally { event.target.value = ''; }
        };
        reader.onerror = (err_reader) => { console.error("アノテーション読込エラー:", err_reader); alert("アノテーション読込失敗"); event.target.value = ''; };
        reader.readAsText(file);
    }

    // --- Event Listener Registration for Undo/Redo ---
    undoBtn.addEventListener('click', undo);
    redoBtn.addEventListener('click', redo);

    function loadInitialSettings() {
        setDrawingMode('select');
    }
}); // End DOMContentLoaded