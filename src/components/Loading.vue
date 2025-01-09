<template>
  <div class="absolute inset-0 flex items-center justify-center">
    <div class="flex">
      <div class="loading" v-for="(text, index) in '模型加载中...'" :key="index">
        {{ text }}
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">

</script>

<style scoped>
.loading {
  font-size: 24px;
  font-weight: bold;
  color: #1abc9c;
  animation: bounce 2s ease-in-out infinite;
}

.loading:nth-child(2) {
  animation-delay: 0.2s;
}

.loading:nth-child(3) {
  animation-delay: 0.3s;
}

.loading:nth-child(4) {
  animation-delay: 0.4s;
}

.loading:nth-child(5) {
  animation-delay: 0.5s;
}

.loading:nth-child(6) {
  animation-delay: 0.6s;
}

.loading:nth-child(7) {
  animation-delay: 0.7s;
}

.loading:nth-child(8) {
  animation-delay: 0.8s;
}

.loading:nth-child(9) {
  animation-delay: 0.9s;
}

.loading:nth-child(10) {
  animation-delay: 1s;
}

@keyframes bounce {
  0% {
    transform: translate3d(0, 0, 0) rotateY(0deg);
  }

  50% {
    transform: translate3d(0, -20px, 0) rotateY(0deg);
  }

  100% {
    transform: translate3d(0, 0, 0) rotateY(0deg);
  }
}
</style>