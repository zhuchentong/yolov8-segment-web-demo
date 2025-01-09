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
const COLORS = Array.from({ length: 32 }, (_, i) => {
  // 使用HSL颜色空间，均匀分布色相
  const hue = (i * 360 / 32) // 色相均匀分布在0-360度
  const saturation = 80 // 固定饱和度为80%
  const lightness = 60  // 固定亮度为60%
  
  // 将HSL转换为RGB
  const h = hue / 360
  const s = saturation / 100
  const l = lightness / 100

  let r: number, g: number, b: number

  if (s === 0) {
    r = g = b = Math.round(l * 255)
  } else {
    const hue2rgb = (p: number, q: number, t: number) => {
      if (t < 0) t += 1
      if (t > 1) t -= 1
      if (t < 1/6) return p + (q - p) * 6 * t
      if (t < 1/2) return q
      if (t < 2/3) return p + (q - p) * (2/3 - t) * 6
      return p
    }

    const q = l < 0.5 ? l * (1 + s) : l + s - l * s
    const p = 2 * l - q

    r = Math.round(hue2rgb(p, q, h + 1/3) * 255)
    g = Math.round(hue2rgb(p, q, h) * 255)
    b = Math.round(hue2rgb(p, q, h - 1/3) * 255)
  }

  return { r, g, b }
})

// 从标签中提取颜色索引
function getLabelIndex(label: string): number {
  // 假设标签格式为 "T11" 或 "T1_1"
  const matches = label.match(/T(\d+)_?(\d+)?/)
  if (!matches) return 0

  const row = parseInt(matches[1])
  const col = matches[2] ? parseInt(matches[2]) : 1
  
  // 计算索引 (row-1)*8 + (col-1)
  const index = (row - 1) * 8 + (col - 1)
  return Math.min(index, 31) // 确保索引不超过31
}

export function drawBoxes(boxes: DetectionBox[], ctx: CanvasRenderingContext2D) {
  const canvas = ctx.canvas
  
  boxes.forEach((box,index) => {
    const { x1, y1, x2, y2, maskArray, label, confidence } = box
    const boxWidth = Math.round(x2 - x1)
    const boxHeight = Math.round(y2 - y1)

    const color = COLORS[index]

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
          } else {
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
