import cv from '@techstark/opencv-js'

async function createInputSource(image: ArrayBuffer): Promise<cv.Mat> {
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d')
  const blob = new Blob([image], { type: 'image/png' })
  const bitmap = await createImageBitmap(blob)

  canvas.width = bitmap.width
  canvas.height = bitmap.height
  ctx?.drawImage(bitmap, 0, 0, canvas.width, canvas.height)

  return cv.imread(canvas)
}

function resizeInputSource(data: cv.Mat) {
  const maxSize = Math.max(data.rows, data.cols) // get max size from width and height
  const xPad = maxSize - data.cols // set xPadding
  const yPad = maxSize - data.rows // set yPadding
  const matPad = new cv.Mat() // new mat for padded image
  cv.copyMakeBorder(data, matPad, 0, yPad, 0, xPad, cv.BORDER_CONSTANT) // padding black

  const output = new cv.Mat()
  cv.resize(matPad, output, new cv.Size(640, 640), 0, 0, cv.INTER_AREA)

  return output
}

// function drawInputSource(data: cv.Mat, canvas: HTMLCanvasElement) {
//   canvas.height = data.rows
//   canvas.width = data.cols
//   cv.imshow(canvas, data)
// }

function formatInputData(data: cv.Mat) {
  const canvas = document.createElement('canvas')
  const context = canvas.getContext('2d')
  cv.imshow(canvas, data)

  const imageData = context!.getImageData(0, 0, 640, 640)
  const source = imageData.data

  // 创建一个 Float32Array 来存储数据
  // NCHW 格式：1 (batch) x 3 (channels) x 640 (height) x 640 (width)
  const inputTensor = new Float32Array(1 * 3 * 640 * 640)

  // 重组数据为 NCHW 格式
  for (let c = 0; c < 3; c++) { // channels
    for (let h = 0; h < 640; h++) { // height
      for (let w = 0; w < 640; w++) { // width
        const pixelIndex = (h * 640 + w) * 4 // RGBA 格式，每个像素4个值
        // 计算目标索引：((c * H * W) + (h * W) + w)
        const tensorIndex = c * 640 * 640 + h * 640 + w
        // 归一化像素值到 [0,1]
        inputTensor[tensorIndex] = source[pixelIndex + c] / 255.0
      }
    }
  }

  // 直接返回 Float32Array
  return inputTensor
}

export async function processInput(image: ArrayBuffer) {
  // 创建input
  const input = await createInputSource(image)
  // 处理inputƒ
  const data = await resizeInputSource(input)
  // output写入canvas
  // drawInputSource(data, canvas)

  return formatInputData(data)
}
