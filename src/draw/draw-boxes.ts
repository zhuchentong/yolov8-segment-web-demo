import { generateColors } from '../common/color'

interface DetectionBox {
  x1: number
  y1: number
  x2: number
  y2: number
  label: string
  confidence: number
  maskArray: number[]
}

// 生成32种不同的颜色
const COLORS = generateColors(32)

function getLabelColor(label: string) {
  const value = Number(label.replace('T', ''))
  const key = (Math.floor(value / 10) - 1) * 8 + value % 10
  return COLORS[key]
}

export function drawBoxes(boxes: DetectionBox[], ctx: CanvasRenderingContext2D) {
  boxes.forEach((box) => {
    const { x1, y1, x2, y2, maskArray, label } = box
    const boxWidth = Math.round(x2 - x1)
    const boxHeight = Math.round(y2 - y1)

    const color = getLabelColor(box.label)

    // 创建掩码canvas
    const maskCanvas = document.createElement('canvas')
    maskCanvas.width = boxWidth
    maskCanvas.height = boxHeight
    const maskCtx = maskCanvas.getContext('2d')!

    const maskImageData = maskCtx.createImageData(boxWidth, boxHeight)
    const maskData = maskImageData.data

    // 将mask值映射到图像区域
    for (let y = 0; y < boxHeight; y++) {
      for (let x = 0; x < boxWidth; x++) {
        const dataIndex = (y * boxWidth + x) * 4
        const maskIndex = y * boxWidth + x

        if (maskIndex < maskArray.length) {
          const maskValue = maskArray[maskIndex]
          if (maskValue > 0.7) {
            maskData[dataIndex] = color.r
            maskData[dataIndex + 1] = color.g
            maskData[dataIndex + 2] = color.b
            maskData[dataIndex + 3] = Math.floor(maskValue * 255)
          }
          else {
            maskData[dataIndex + 3] = 0
          }
        }
      }
    }

    // 将掩码数据绘制到临时canvas上
    maskCtx.putImageData(maskImageData, 0, 0)

    // 将掩码绘制到原始canvas上
    ctx.save()
    ctx.globalAlpha = 0.5
    ctx.drawImage(maskCanvas, x1, y1, boxWidth, boxHeight)

    // 计算中心点坐标
    const centerX = x1 + boxWidth / 2
    const centerY = y1 + boxHeight / 2

    // 绘制标签文字
    ctx.globalAlpha = 1.0
    ctx.font = '10px Arial'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'

    // 绘制文字背景
    const text = `${label}`

    // 绘制文字
    ctx.fillStyle = '#000000'
    ctx.fillText(text, centerX, centerY)

    ctx.restore()
  })
}
