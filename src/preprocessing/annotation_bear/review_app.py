"""
Simple Web-based Review Interface for Multi-Model Annotations

This Flask app provides a simple UI to review annotations that need human verification.

Usage:
    python -m src.preprocessing.review_app --review-queue data/review_queue/

Then open http://localhost:5000 in your browser.
"""

import argparse
from pathlib import Path
import json
from flask import Flask, render_template_string, request, jsonify, send_from_directory
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

app = Flask(__name__)

# Global variables
REVIEW_QUEUE_DIR = None
CURRENT_INDEX = 0
REVIEW_ITEMS = []

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Annotation Review Interface</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .header {
            background: #2c3e50;
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .header h1 {
            margin: 0 0 10px 0;
        }
        .progress {
            background: #34495e;
            border-radius: 4px;
            padding: 10px;
            margin-top: 10px;
        }
        .container {
            background: white;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .image-container {
            text-align: center;
            margin-bottom: 20px;
        }
        .image-container img {
            max-width: 100%;
            max-height: 600px;
            border: 2px solid #ddd;
            border-radius: 4px;
        }
        .info-box {
            background: #ecf0f1;
            padding: 15px;
            border-radius: 4px;
            margin-bottom: 20px;
        }
        .model-results {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .model-card {
            background: #f8f9fa;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
        }
        .model-card h3 {
            margin-top: 0;
            color: #2c3e50;
        }
        .detection-item {
            background: white;
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
            border-left: 4px solid #3498db;
        }
        .controls {
            display: flex;
            gap: 10px;
            margin-top: 20px;
            flex-wrap: wrap;
        }
        .btn {
            padding: 12px 24px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            transition: background 0.3s;
        }
        .btn-primary {
            background: #3498db;
            color: white;
        }
        .btn-primary:hover {
            background: #2980b9;
        }
        .btn-success {
            background: #27ae60;
            color: white;
        }
        .btn-success:hover {
            background: #229954;
        }
        .btn-warning {
            background: #f39c12;
            color: white;
        }
        .btn-warning:hover {
            background: #e67e22;
        }
        .btn-secondary {
            background: #95a5a6;
            color: white;
        }
        .btn-secondary:hover {
            background: #7f8c8d;
        }
        .review-reason {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 12px;
            margin-bottom: 20px;
            border-radius: 4px;
        }
        .no-items {
            text-align: center;
            padding: 40px;
            color: #7f8c8d;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Annotation Review Interface</h1>
        <p>Review annotations that need human verification</p>
        <div class="progress" id="progress">
            Loading...
        </div>
    </div>

    <div class="container" id="main-container">
        <div class="no-items">Loading review items...</div>
    </div>

    <script>
        let currentIndex = 0;
        let reviewItems = [];

        async function loadReviewItems() {
            const response = await fetch('/api/review-items');
            const data = await response.json();
            reviewItems = data.items;
            currentIndex = 0;
            displayCurrentItem();
        }

        function displayCurrentItem() {
            if (reviewItems.length === 0) {
                document.getElementById('main-container').innerHTML = `
                    <div class="no-items">
                        <h2>✓ All Done!</h2>
                        <p>No items need review at the moment.</p>
                    </div>
                `;
                document.getElementById('progress').innerHTML = 'No items to review';
                return;
            }

            const item = reviewItems[currentIndex];
            const progress = `Item ${currentIndex + 1} of ${reviewItems.length}`;
            document.getElementById('progress').innerHTML = progress;

            let modelResultsHTML = '';
            for (const [modelName, detections] of Object.entries(item.detections_by_model)) {
                let detectionsHTML = '';
                if (detections.length === 0) {
                    detectionsHTML = '<p style="color: #95a5a6;">No detections</p>';
                } else {
                    detectionsHTML = detections.map((det, idx) => `
                        <div class="detection-item">
                            <strong>Detection ${idx + 1}</strong><br>
                            Label: ${det.label}<br>
                            Score: ${det.score.toFixed(3)}<br>
                            Box: [${det.box.map(v => v.toFixed(1)).join(', ')}]
                        </div>
                    `).join('');
                }

                modelResultsHTML += `
                    <div class="model-card">
                        <h3>${modelName.toUpperCase()}</h3>
                        <p><strong>${detections.length}</strong> detection(s)</p>
                        ${detectionsHTML}
                    </div>
                `;
            }

            document.getElementById('main-container').innerHTML = `
                <div class="info-box">
                    <strong>Image:</strong> ${item.filename}<br>
                    <strong>Prompt:</strong> ${item.text_prompt}
                </div>

                <div class="review-reason">
                    <strong>⚠️  Review Reason:</strong> ${item.review_reason}
                </div>

                <div class="image-container">
                    <img src="/api/image/${encodeURIComponent(item.filename)}" alt="Review image">
                </div>

                <h2>Model Detections</h2>
                <div class="model-results">
                    ${modelResultsHTML}
                </div>

                <div class="controls">
                    <button class="btn btn-success" onclick="approve()">✓ Approve All Detections</button>
                    <button class="btn btn-warning" onclick="approveNone()">✗ No Valid Detections</button>
                    <button class="btn btn-secondary" onclick="skip()">⏭ Skip (Review Later)</button>
                    <button class="btn btn-primary" onclick="next()" style="margin-left: auto;">Next →</button>
                </div>
            `;
        }

        async function approve() {
            await saveDecision('approved');
        }

        async function approveNone() {
            await saveDecision('rejected');
        }

        async function skip() {
            next();
        }

        async function saveDecision(decision) {
            const item = reviewItems[currentIndex];
            await fetch('/api/save-decision', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    filename: item.filename,
                    decision: decision
                })
            });
            next();
        }

        function next() {
            currentIndex++;
            if (currentIndex >= reviewItems.length) {
                currentIndex = reviewItems.length;
                document.getElementById('main-container').innerHTML = `
                    <div class="no-items">
                        <h2>✓ Review Complete!</h2>
                        <p>You've reviewed all items in the queue.</p>
                        <button class="btn btn-primary" onclick="location.reload()">Reload</button>
                    </div>
                `;
                document.getElementById('progress').innerHTML = 'Review complete';
            } else {
                displayCurrentItem();
            }
        }

        // Load items on page load
        window.onload = loadReviewItems;
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Serve the main review interface."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/review-items')
def get_review_items():
    """Get list of items that need review."""
    global REVIEW_QUEUE_DIR, REVIEW_ITEMS

    detections_dir = REVIEW_QUEUE_DIR / "detections"

    if not detections_dir.exists():
        return jsonify({"items": []})

    items = []
    for json_file in sorted(detections_dir.glob("*.json")):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                items.append({
                    "filename": Path(data["image_path"]).name,
                    "text_prompt": data.get("text_prompt", ""),
                    "detections_by_model": data.get("detections_by_model", {}),
                    "review_reason": data.get("review_reason", "Unknown")
                })
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue

    REVIEW_ITEMS = items
    return jsonify({"items": items})


@app.route('/api/image/<filename>')
def get_image(filename):
    """Serve an image from the review queue."""
    global REVIEW_QUEUE_DIR
    images_dir = REVIEW_QUEUE_DIR / "images"
    return send_from_directory(images_dir, filename)


@app.route('/api/save-decision', methods=['POST'])
def save_decision():
    """Save a review decision."""
    global REVIEW_QUEUE_DIR

    data = request.json
    filename = data.get('filename')
    decision = data.get('decision')

    # Save decision to a separate file
    decisions_dir = REVIEW_QUEUE_DIR / "decisions"
    decisions_dir.mkdir(exist_ok=True)

    decision_file = decisions_dir / f"{Path(filename).stem}_decision.json"
    with open(decision_file, 'w') as f:
        json.dump({
            "filename": filename,
            "decision": decision,
        }, f, indent=2)

    return jsonify({"status": "success"})


def main():
    parser = argparse.ArgumentParser(
        description="Web-based review interface for multi-model annotations"
    )
    parser.add_argument(
        "--review-queue", "-r",
        required=True,
        help="Path to review queue directory"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=5000,
        help="Port to run the server on (default: 5000)"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to run the server on (default: 127.0.0.1)"
    )

    args = parser.parse_args()

    global REVIEW_QUEUE_DIR
    REVIEW_QUEUE_DIR = Path(args.review_queue)

    if not REVIEW_QUEUE_DIR.exists():
        print(f"Error: Review queue directory not found: {REVIEW_QUEUE_DIR}")
        return

    print(f"\n{'='*60}")
    print("Starting Annotation Review Server")
    print(f"{'='*60}")
    print(f"Review queue: {REVIEW_QUEUE_DIR}")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"{'='*60}\n")
    print("Open the URL in your browser to start reviewing.")
    print("Press Ctrl+C to stop the server.\n")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
