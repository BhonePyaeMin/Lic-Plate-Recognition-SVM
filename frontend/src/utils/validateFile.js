/** Client-side pre-check. The backend repeats and enforces every rule. */
export function validateFile(file, limits) {
  const ext = file.name.includes('.') ? file.name.split('.').pop().toLowerCase() : ''
  if (!limits.allowedExtensions.includes(ext)) {
    return {
      code: 'UNSUPPORTED_FILE_TYPE',
      message: `This file type is not supported. Choose a ${limits.allowedExtensions.map((e) => e.toUpperCase()).join(', ')} image.`,
    }
  }
  if (file.size === 0) {
    return { code: 'EMPTY_FILE', message: 'This file is empty. Choose a different image.' }
  }
  if (file.size > limits.maxUploadMb * 1024 * 1024) {
    return {
      code: 'FILE_TOO_LARGE',
      message: `This image is larger than ${limits.maxUploadMb} MB. Resize or compress it and try again.`,
    }
  }
  return null
}

export function formatBytes(n) {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`
  return `${(n / 1024 / 1024).toFixed(1)} MB`
}
