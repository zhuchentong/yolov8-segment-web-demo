export function generateColors(count: number) {
  return Array.from({ length: count }, (_, i) => {
    // 使用HSL颜色空间，均匀分布色相
    const hue = (i * 360 / 32) // 色相均匀分布在0-360度
    const saturation = 80 // 固定饱和度为80%
    const lightness = 60 // 固定亮度为60%

    // 将HSL转换为RGB
    const h = hue / 360
    const s = saturation / 100
    const l = lightness / 100

    let r: number, g: number, b: number

    if (s === 0) {
      r = g = b = Math.round(l * 255)
    }
    else {
      const hue2rgb = (p: number, q: number, t: number) => {
        if (t < 0)
          t += 1
        if (t > 1)
          t -= 1
        if (t < 1 / 6)
          return p + (q - p) * 6 * t
        if (t < 1 / 2)
          return q
        if (t < 2 / 3)
          return p + (q - p) * (2 / 3 - t) * 6
        return p
      }

      const q = l < 0.5 ? l * (1 + s) : l + s - l * s
      const p = 2 * l - q

      r = Math.round(hue2rgb(p, q, h + 1 / 3) * 255)
      g = Math.round(hue2rgb(p, q, h) * 255)
      b = Math.round(hue2rgb(p, q, h - 1 / 3) * 255)
    }

    return { r, g, b }
  })
}
