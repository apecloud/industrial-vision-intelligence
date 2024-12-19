docker buildx build \
  --platform linux/amd64 \
  -t yolov8:2.0 \
  --load \
  .
