import type ort from 'onnxruntime-web'

// YOLOv8可以检测的80个类别标签
const YOLO_CLASSES = [
  'person',
  'bicycle',
  'car',
  'motorcycle',
  'airplane',
  'bus',
  'train',
  'truck',
  'boat',
  'traffic light',
  'fire hydrant',
  'stop sign',
  'parking meter',
  'bench',
  'bird',
  'cat',
  'dog',
  'horse',
  'sheep',
  'cow',
  'elephant',
  'bear',
  'zebra',
  'giraffe',
  'backpack',
  'umbrella',
  'handbag',
  'tie',
  'suitcase',
  'frisbee',
  'skis',
  'snowboard',
  'sports ball',
  'kite',
  'baseball bat',
  'baseball glove',
  'skateboard',
  'surfboard',
  'tennis racket',
  'bottle',
  'wine glass',
  'cup',
  'fork',
  'knife',
  'spoon',
  'bowl',
  'banana',
  'apple',
  'sandwich',
  'orange',
  'broccoli',
  'carrot',
  'hot dog',
  'pizza',
  'donut',
  'cake',
  'chair',
  'couch',
  'potted plant',
  'bed',
  'dining table',
  'toilet',
  'tv',
  'laptop',
  'mouse',
  'remote',
  'keyboard',
  'cell phone',
  'microwave',
  'oven',
  'toaster',
  'sink',
  'refrigerator',
  'book',
  'clock',
  'vase',
  'scissors',
  'teddy bear',
  'hair drier',
  'toothbrush',
] as const

interface DetectionBox {
  x1: number
  y1: number
  x2: number
  y2: number
  label: typeof YOLO_CLASSES[number]
  confidence: number
  mask: number[][]
}

export function processOutput(
  outputs: [ort.Tensor, ort.Tensor],
  imgWidth: number,
  imgHeight: number,
): DetectionBox[] {
  const [output0, output1] = outputs
  const outputArray = Array.from(output0.data as Float32Array)
  const maskArray = Array.from(output1.data as Float32Array)

  // 提取边界框和类别信息
  const boxesOutput: number[][] = []
  for (let i = 0; i < 8400; i++) {
    boxesOutput.push(outputArray.slice(i * 84, (i + 1) * 84))
  }

  // 处理分割掩码
  const masksOutput = Array.from({ length: 32 }).fill(0).map((_, i) =>
    Array.from({ length: 160 * 160 }).fill(0).map((_, j) => maskArray[i * 160 * 160 + j]),
  )
  const masksProto = outputArray.slice(8400 * 84).reduce((acc, val, i) => {
    const maskIdx = Math.floor(i / 32)
    const protoIdx = i % 32
    if (!acc[maskIdx])
      acc[maskIdx] = []
    acc[maskIdx][protoIdx] = val
    return acc
  }, [] as number[][])

  let boxes: DetectionBox[] = []

  // 处理每个检测结果
  for (let i = 0; i < boxesOutput.length; i++) {
    const row = boxesOutput[i]

    // 获取最高置信度的类别
    const scores = row.slice(4, 84)
    const maxScore = Math.max(...scores)
    const classId = scores.indexOf(maxScore)

    // 过滤低置信度的检测结果
    if (maxScore < 0.5)
      continue

    // 将边界框坐标转换回原始图像尺寸
    const xc = (row[0] / 640.0) * imgWidth
    const yc = (row[1] / 640.0) * imgHeight
    const w = (row[2] / 640.0) * imgWidth
    const h = (row[3] / 640.0) * imgHeight
    const x1 = xc - w / 2
    const y1 = yc - h / 2
    const x2 = xc + w / 2
    const y2 = yc + h / 2

    // 处理掩码
    const mask = processMask(
      masksOutput[i],
      { x1, y1, x2, y2 },
      imgWidth,
      imgHeight,
    )

    boxes.push({
      x1,
      y1,
      x2,
      y2,
      label: YOLO_CLASSES[classId],
      confidence: maxScore,
      mask,
    })
  }

  // 按置信度排序
  boxes.sort((a, b) => b.confidence - a.confidence)

  // 执行非极大值抑制(NMS)
  const result: DetectionBox[] = []
  while (boxes.length > 0) {
    result.push(boxes[0])
    boxes = boxes.filter(box => iou(boxes[0], box) < 0.7)
  }

  return result
}

function processMask(
  mask: number[],
  rect: { x1: number, y1: number, x2: number, y2: number },
  imgWidth: number,
  imgHeight: number,
): number[][] {
  const { x1, y1, x2, y2 } = rect
  const maskSize = 160
  const result = Array.from<number>({ length: Math.round(y2 - y1) })
    .fill(0)
    .map(() => Array.from<number>({ length: Math.round(x2 - x1) }).fill(0))

  // 将掩码值应用到对应区域
  for (let y = 0; y < result.length; y++) {
    for (let x = 0; x < result[0].length; x++) {
      const maskX = Math.round((x / (x2 - x1)) * maskSize)
      const maskY = Math.round((y / (y2 - y1)) * maskSize)
      result[y][x] = mask[maskY * maskSize + maskX] > 0 ? 255 : 0
    }
  }

  return result
}

function iou(box1: DetectionBox, box2: DetectionBox): number {
  return intersection(box1, box2) / union(box1, box2)
}

function union(box1: DetectionBox, box2: DetectionBox): number {
  const box1Area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
  const box2Area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
  return box1Area + box2Area - intersection(box1, box2)
}

function intersection(box1: DetectionBox, box2: DetectionBox): number {
  const x1 = Math.max(box1.x1, box2.x1)
  const y1 = Math.max(box1.y1, box2.y1)
  const x2 = Math.min(box1.x2, box2.x2)
  const y2 = Math.min(box1.y2, box2.y2)
  return Math.max(0, x2 - x1) * Math.max(0, y2 - y1)
}
