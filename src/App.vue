<script setup lang="ts">
import ort from 'onnxruntime-web'
import { onMounted, ref, useTemplateRef } from 'vue'
import { download } from './common/download'
import Loading from './components/Loading.vue'
import { drawBoxes } from './draw/draw-boxes'
import { processInput } from './process/input'
import { processOutput } from './process/output'
import { useFileDialog } from '@vueuse/core'

const canvas = useTemplateRef('canvas')
const loaded = ref(false)

const dialogImage = useFileDialog({
  accept: 'image/*'
})


let session: ort.InferenceSession
let inputBuffer: ArrayBuffer

async function loadedModel() {
  // 下载模型文件
  const buffer = await download('/models/best.onnx')
  // 加载模型
  session = await ort.InferenceSession.create(buffer)
  // 模型加载完成
  loaded.value = true
}

async function loadImage() {
  // 下载模型文件
  inputBuffer = await download('/images/image.png')
}

async function runModel() {
  if (inputBuffer && session) {
    const input = await processInput(inputBuffer)

    const tensor = new ort.Tensor(input, [1, 3, 640, 640])
    const output = await session.run({
      images: tensor,
    })


    const boxes = await processOutput(output, 640, 640)
    console.log(boxes)
    canvas.value!.width = 640
    canvas.value!.height = 640
    drawBoxes(boxes, canvas.value!.getContext('2d')!)
  }
}


dialogImage.onChange(async (files) => {
  if (files) {
    const [file] = files
    inputBuffer = await file.arrayBuffer()
    const ctx = canvas.value!.getContext('2d')!
    ctx.clearRect(0,0,640,640)

    runModel()
  }
})

onMounted(async () => {
  await loadedModel()
  await loadImage()

  runModel()
})
</script>

<template>
  <main v-if="loaded" class="absolute inset-0 flex justify-center items-center">
    <div class="w-640px overflow-hidden relative">
      <img src="/images/image.png" class="w-full h-auto"/>
      <canvas ref="canvas" class="absolute left-0 top-0" />
    </div>
  <button type="button" @click="()=>dialogImage.open()" class="absolute bottom-50px right-50px width-80px height-80px rounded-full">上传图片</button>
  </main>
  <Loading v-else />
</template>

<style scoped>

</style>
