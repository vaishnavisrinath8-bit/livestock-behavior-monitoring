# Model Weights Directory

This directory stores deep learning model weights used by the Livestock Monitoring System.

## Pretrained Detection Model

- **File**: `yolov8n.pt`
- **Source**: Ultralytics YOLOv8 (COCO dataset pretrained)
- **Class used**: `cow` (COCO class ID 19)
- **Note**: Ultralytics automatically downloads this file on first use if internet access is present.

## Custom Behavior Classification Model (Optional)

- **Expected File**: `behavior_model.pt`
- **Classes**:
  1. `Standing`
  2. `Feeding`
  3. `Lying/Resting`
  4. `Unknown`
- **Status**: When not present, the system automatically runs in **Baseline Estimation Mode** with transparent heuristic indicators, returning `Unknown` when uncertain rather than hallucinating predictions.
