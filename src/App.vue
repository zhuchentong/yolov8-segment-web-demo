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
const loading = ref(false)
const dialogImage = useFileDialog({
  accept: 'image/*'
})

const images = [
  '/images/01.png',
  '/images/02.png',
  '/images/03.png',
  '/images/04.png',
  '/images/05.png',
  '/images/06.png',
  '/images/07.png',
  '/images/08.png',
  '/images/09.png',
  '/images/10.png',
]

const image = ref(images[0])


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
  inputBuffer = await download(image.value)
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
    ctx.clearRect(0, 0, 640, 640)

    runModel()
  }
})

async function onRender(url: string) {
  loading.value = true

  try {
    image.value = url
    const ctx = canvas.value?.getContext('2d')
    ctx?.clearRect(0, 0, 640, 640)

    await loadImage()
    await runModel()
  } catch (ex) {
    console.error(ex)
  } finally {
    loading.value = false
  }

}

onMounted(async () => {
  await loadedModel()

  onRender(image.value)
})
</script>

<template>
  <main v-if="loaded" class="absolute inset-0 flex flex-col">
    <h1 class="absolute left-20px top-10px text-#fff z-10">FDI牙位标识</h1>
    <div class="flex-auto relative">
      <div class="absolute inset-0 flex justify-center items-center bg-#333">
        <div class="w-640px overflow-hidden relative">
          <img :src="image" class="w-full h-auto" />
          <canvas ref="canvas" class="absolute left-0 top-0" />
        </div>
      </div>
    </div>
    <div class="py-2 overflow-auto bg-#fff">
      <div class="flex items-center image-list">
        <div @click="() => onRender(url)" v-for="url in images" :key="url"
          class="image-item cursor-pointer text-0 px-1">
          <img :class="{ active: image === url }" :src="url" class="w-200px h-auto">
        </div>
      </div>
    </div>
  </main>
  <Loading v-else />
  <div v-if="loading" class="absolute inset-0 z-100 bg-#000A text-#fff flex-col flex  space-y-4 items-center justify-center">
    <!-- <svg aria-hidden="true" class="w-8 h-8 text-gray-200 animate-spin dark:text-gray-600 fill-blue-600" viewBox="0 0 100 101" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M100 50.5908C100 78.2051 77.6142 100.591 50 100.591C22.3858 100.591 0 78.2051 0 50.5908C0 22.9766 22.3858 0.59082 50 0.59082C77.6142 0.59082 100 22.9766 100 50.5908ZM9.08144 50.5908C9.08144 73.1895 27.4013 91.5094 50 91.5094C72.5987 91.5094 90.9186 73.1895 90.9186 50.5908C90.9186 27.9921 72.5987 9.67226 50 9.67226C27.4013 9.67226 9.08144 27.9921 9.08144 50.5908Z" fill="currentColor"/>
        <path d="M93.9676 39.0409C96.393 38.4038 97.8624 35.9116 97.0079 33.5539C95.2932 28.8227 92.871 24.3692 89.8167 20.348C85.8452 15.1192 80.8826 10.7238 75.2124 7.41289C69.5422 4.10194 63.2754 1.94025 56.7698 1.05124C51.7666 0.367541 46.6976 0.446843 41.7345 1.27873C39.2613 1.69328 37.813 4.19778 38.4501 6.62326C39.0873 9.04874 41.5694 10.4717 44.0505 10.1071C47.8511 9.54855 51.7191 9.52689 55.5402 10.0491C60.8642 10.7766 65.9928 12.5457 70.6331 15.2552C75.2735 17.9648 79.3347 21.5619 82.5849 25.841C84.9175 28.9121 86.7997 32.2913 88.1811 35.8758C89.083 38.2158 91.5421 39.6781 93.9676 39.0409Z" fill="currentFill"/>
    </svg> -->
    <div>牙位标识中...</div>
  </div>
</template>

<style scoped>
.image-list img {
  border: solid 2px #000;
  border-radius: 5px;
  overflow: hidden;
}

.image-list img.active {
  border-color: red;
}
</style>
