document.addEventListener('DOMContentLoaded', function() {
    const canvas = document.getElementById('diagnosisCanvas');
    const ctx = canvas.getContext('2d');
    const offscreenCanvas = document.createElement('canvas');
    const offscreenCtx = offscreenCanvas.getContext('2d');

    // Initialize offscreen canvas to match canvas size
    offscreenCanvas.width = canvas.width;
    offscreenCanvas.height = canvas.height;

    let isDrawing = false;
    let startX, startY;
    let currentTool = 'brush';
    let selectedColor = '#000000';
    let selectedSize = 5;

    // Function to resize canvas while keeping the drawing
    function resizeCanvas() {
        // Scale factor based on new dimensions
        const scaleX = canvas.clientWidth / canvas.width;
        const scaleY = canvas.clientHeight / canvas.height;

        // Resize the offscreen canvas content
        canvas.width = canvas.clientWidth;
        canvas.height = canvas.clientHeight;

        // Redraw content with scaling
        ctx.scale(scaleX, scaleY);
        ctx.drawImage(offscreenCanvas, 0, 0);
        ctx.scale(1 / scaleX, 1 / scaleY);  // Reset scale for future strokes
    }

    // Tool selection and drawing functionality
    function draw(e) {
        if (currentTool === 'brush') {
            ctx.lineWidth = selectedSize;
            ctx.lineCap = 'round';
            ctx.strokeStyle = selectedColor;
            ctx.beginPath();
            ctx.moveTo(startX, startY);
            ctx.lineTo(e.offsetX, e.offsetY);
            ctx.stroke();
            offscreenCtx.lineWidth = selectedSize;
            offscreenCtx.lineCap = 'round';
            offscreenCtx.strokeStyle = selectedColor;
            offscreenCtx.beginPath();
            offscreenCtx.moveTo(startX, startY);
            offscreenCtx.lineTo(e.offsetX, e.offsetY);
            offscreenCtx.stroke();
            startX = e.offsetX;
            startY = e.offsetY;
        }
    }

    // Mouse events for drawing
    canvas.addEventListener('mousedown', (e) => {
        isDrawing = true;
        startX = e.offsetX;
        startY = e.offsetY;
    });

    canvas.addEventListener('mousemove', (e) => {
        if (isDrawing) draw(e);
    });

    canvas.addEventListener('mouseup', () => (isDrawing = false));

    // Resize event listener
    window.addEventListener('resize', resizeCanvas);
});
