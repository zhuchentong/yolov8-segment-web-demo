export function download(url: string): Promise<ArrayBuffer> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest()
    request.open('GET', url, true)
    request.responseType = 'arraybuffer'
    request.onload = function () {
      if (this.status >= 200 && this.status < 300) {
        resolve(request.response)
      }
      else {
        reject(new Error(request.statusText))
      }

      resolve(request.response)
    }

    request.onerror = function () {
      reject(new Error(request.statusText))
    }

    request.send()
  })
};
