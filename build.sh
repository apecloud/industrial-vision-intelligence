docker buildx build \
  --platform linux/arm64 \
  -t yolov8:1.0 \
  --load \
  .
