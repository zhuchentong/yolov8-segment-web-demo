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
  maskArray: number[]
}

export function processOutput(
  data: ort.InferenceSession.OnnxValueMapType,
  originalWidth: number,
  originalHeight: number,
): DetectionBox[] {
  const { output0, output1 } = data
  const output = output0.data as Float32Array
  const proto = output1.data as Float32Array
  const boxes: DetectionBox[] = []
  const num_masks = 32

  // 获取原型维度
  const [_, protoHeight, protoWidth] = output1.dims.slice(1)
  for (let index = 0; index < 8400; index++) {
    const [class_id, confidence] = [...Array.from({ length: YOLO_CLASSES.length }).keys()]
      .map(col => [col, output[8400 * (col + 4) + index]])
      .reduce((accum, item) => (item[1] > accum[1] ? item : accum), [0, 0])

    if (confidence < 0.05) {
      continue
    }

    const xc = output[index]
    const yc = output[8400 + index]
    const w = output[2 * 8400 + index]
    const h = output[3 * 8400 + index]
    const x1 = ((xc - w / 2) / 640) * originalWidth
    const y1 = ((yc - h / 2) / 640) * originalHeight
    const x2 = ((xc + w / 2) / 640) * originalWidth
    const y2 = ((yc + h / 2) / 640) * originalHeight

    // 获取 mask 系数
    const maskOffset = 8400 * (YOLO_CLASSES.length + 4)
    const maskCoeffs = new Float32Array(num_masks)
    for (let i = 0; i < num_masks; i++) {
      maskCoeffs[i] = output[maskOffset + index + i * 8400]
    }

    // 计算完整的mask
    const fullMask = new Float32Array(protoHeight * protoWidth)

    // 对每个像素位置计算mask值
    for (let h = 0; h < protoHeight; h++) {
      for (let w = 0; w < protoWidth; w++) {
        let sum = 0
        // 对每个通道进行计算
        for (let c = 0; c < num_masks; c++) {
          const protoIndex = c * protoHeight * protoWidth + h * protoWidth + w
          sum += maskCoeffs[c] * proto[protoIndex]
        }
        // 应用sigmoid激活函数
        fullMask[h * protoWidth + w] = 1 / (1 + Math.exp(-sum))
      }
    }

    // 1. 计算检测框在160x160 mask中的位置
    const maskWidth = 160
    const maskHeight = 160

    // 将边界框坐标映射到160x160的mask空间
    const maskX1 = Math.floor((x1 / originalWidth) * maskWidth)
    const maskY1 = Math.floor((y1 / originalHeight) * maskHeight)
    const maskX2 = Math.ceil((x2 / originalWidth) * maskWidth)
    const maskY2 = Math.ceil((y2 / originalHeight) * maskHeight)

    // 2. 提取对应区域的mask
    const boxMask = new Float32Array((maskX2 - maskX1) * (maskY2 - maskY1))
    let idx = 0

    for (let y = maskY1; y < maskY2; y++) {
      for (let x = maskX1; x < maskX2; x++) {
        if (x >= 0 && x < maskWidth && y >= 0 && y < maskHeight) {
          boxMask[idx] = fullMask[y * maskWidth + x]
        }
        idx++
      }
    }

    // 3. 将mask缩放到实际检测框大小（可选）
    const actualWidth = Math.round(x2 - x1)
    const actualHeight = Math.round(y2 - y1)
    const scaledMask = resizeMask(boxMask, maskX2 - maskX1, maskY2 - maskY1, actualWidth, actualHeight)

    boxes.push({
      x1,
      y1,
      x2,
      y2,
      label: YOLO_CLASSES[class_id],
      confidence,
      maskArray: Array.from(scaledMask),
    })
  }

  // 根据置信度排序
  let boxesWithMasks = [...boxes]
  boxesWithMasks.sort((a, b) => b.confidence - a.confidence)

  const result: DetectionBox[] = []
  while (boxesWithMasks.length > 0) {
    result.push(boxesWithMasks[0])
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

// mask缩放函数
function resizeMask(
  mask: Float32Array,
  fromWidth: number,
  fromHeight: number,
  toWidth: number,
  toHeight: number,
): Float32Array {
  const output = new Float32Array(toWidth * toHeight)

  for (let y = 0; y < toHeight; y++) {
    for (let x = 0; x < toWidth; x++) {
      // 双线性插值
      const srcX = (x / toWidth) * fromWidth
      const srcY = (y / toHeight) * fromHeight

      const x1 = Math.floor(srcX)
      const y1 = Math.floor(srcY)
      const x2 = Math.min(x1 + 1, fromWidth - 1)
      const y2 = Math.min(y1 + 1, fromHeight - 1)

      const xWeight = srcX - x1
      const yWeight = srcY - y1

      const val = mask[y1 * fromWidth + x1] * (1 - xWeight) * (1 - yWeight)
        + mask[y1 * fromWidth + x2] * xWeight * (1 - yWeight)
        + mask[y2 * fromWidth + x1] * (1 - xWeight) * yWeight
        + mask[y2 * fromWidth + x2] * xWeight * yWeight

      output[y * toWidth + x] = val
    }
  }

  return output
}
