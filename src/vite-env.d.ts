/// <reference types="vite/client" />

interface DetectionBox {
  x1: number
  y1: number
  x2: number
  y2: number
  label: typeof YOLO_CLASSES[number]
  confidence: number
  // mask: number[][]
}
