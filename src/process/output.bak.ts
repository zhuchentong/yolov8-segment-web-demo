import type ort from 'onnxruntime-web'

// YOLOv8可以检测的32个类别标签
const YOLO_CLASSES = Array.from({ length: 32 }, (_, i) => `T${Math.ceil((i + 1) / 8)}${(i) % 8 + 1}`)

interface DetectionBox {
  x1: number
  y1: number
  x2: number
  y2: number
  label: typeof YOLO_CLASSES[number]
  confidence: number
}

export function processOutput(
  data: ort.InferenceSession.OnnxValueMapType,
  img_width: number,
  img_height: number,
): DetectionBox[] {
  const { output0 } = data
  const output = output0.data as Float32Array
  const boxes: DetectionBox[] = []
  //   const num_masks = 32 // 根据模型输出的掩码数量

  for (let index = 0; index < 8400; index++) {
    const [class_id, confidence] = [...Array.from({ length: YOLO_CLASSES.length }).keys()]
      .map(col => [col, output[8400 * (col + 4) + index]])
      .reduce((accum, item) => (item[1] > accum[1] ? item : accum), [0, 0])
    console.log(confidence)
    if (confidence < 0.2) {
      continue
    }
    const label = YOLO_CLASSES[class_id]
    // const label = class_id
    const xc = output[index]
    const yc = output[8400 + index]
    const w = output[2 * 8400 + index]
    const h = output[3 * 8400 + index]
    const x1 = ((xc - w / 2) / 640) * img_width
    const y1 = ((yc - h / 2) / 640) * img_height
    const x2 = ((xc + w / 2) / 640) * img_width
    const y2 = ((yc + h / 2) / 640) * img_height

    boxes.push({
      x1,
      y1,
      x2,
      y2,
      label,
      confidence,
    })
  }

  // 根据置信度排序
  let boxesWithMasks = [...boxes]
  boxesWithMasks.sort((a, b) => b.confidence - a.confidence)

  const result: DetectionBox[] = []
  //   const resultMasks = []
  while (boxesWithMasks.length > 0) {
    result.push(boxesWithMasks[0])
    // resultMasks.push(boxesWithMasks[0].mask)
    boxesWithMasks = boxesWithMasks.filter(item => iou(boxesWithMasks[0], item) < 0.7)
  }

  return result
}

function iou(box1: DetectionBox, box2: DetectionBox) {
  return intersection(box1, box2) / union(box1, box2)
}

function intersection(box1: DetectionBox, box2: DetectionBox) {
  const { x1: box1_x1, y1: box1_y1, x2: box1_x2, y2: box1_y2 } = box1
  const { x1: box2_x1, y1: box2_y1, x2: box2_x2, y2: box2_y2 } = box2
  const x1 = Math.max(box1_x1, box2_x1)
  const y1 = Math.max(box1_y1, box2_y1)
  const x2 = Math.min(box1_x2, box2_x2)
  const y2 = Math.min(box1_y2, box2_y2)
  return (x2 - x1) * (y2 - y1)
}

function union(box1: DetectionBox, box2: DetectionBox) {
  const { x1: box1_x1, y1: box1_y1, x2: box1_x2, y2: box1_y2 } = box1
  const { x1: box2_x1, y1: box2_y1, x2: box2_x2, y2: box2_y2 } = box2
  const box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
  const box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
  return box1_area + box2_area - intersection(box1, box2)
}
