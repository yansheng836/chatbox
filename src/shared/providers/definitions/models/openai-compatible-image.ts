import { getOpenAIImageSizeForRatio } from '../image-models'

export function getOpenAIImageGenerationOptions(
  modelId: string,
  aspectRatio?: string
): { size?: `${number}x${number}` } {
  if (!aspectRatio || aspectRatio === 'auto') return {}
  const size = getOpenAIImageSizeForRatio(modelId, aspectRatio)
  return size ? { size } : {}
}

export function compatibleImageApiHeaders(apiHost: string): Record<string, string> | undefined {
  if (apiHost.includes('openrouter.ai')) {
    return {
      'HTTP-Referer': 'https://chatboxai.app',
      'X-Title': 'Chatbox AI',
    }
  }
  if (apiHost.includes('aihubmix.com')) {
    return {
      'APP-Code': 'VAFU9221',
    }
  }
  return undefined
}

export async function generateImageWithOpenAICompatibleAPI(options: {
  apiKey: string
  apiHost: string
  modelId: string
  prompt: string
  n: number
  size?: string
  images?: { imageUrl: string }[]
  signal?: AbortSignal
  fetch?: typeof globalThis.fetch
  headers?: Record<string, string>
}): Promise<string[]> {
  const { apiKey, apiHost, modelId, prompt, n, size, images, signal, fetch: fetchFn, headers } = options

  const resolvedFetch = fetchFn ?? globalThis.fetch
  const resolvedHeaders: Record<string, string> = {
    Authorization: `Bearer ${apiKey}`,
    'Content-Type': 'application/json',
    ...(headers ?? {}),
  }

  const body: Record<string, unknown> = {
    model: modelId,
    prompt,
    n,
    ...(size ? { size } : {}),
    ...(images?.length ? { images: images.map((img) => ({ image_url: img.imageUrl })) } : {}),
  }

  const base = apiHost.replace(/\/+$/, '')
  const url = `${base}/images/generations`

  const response = await resolvedFetch(url, {
    method: 'POST',
    headers: resolvedHeaders,
    body: JSON.stringify(body),
    signal,
  })

  if (!response.ok) {
    let message = `Image generation failed with status ${response.status}`
    try {
      const text = await response.text()
      const errorJson = JSON.parse(text) as { error?: { message?: string }; message?: string }
      if (errorJson?.error?.message) {
        message = errorJson.error.message
      } else if (errorJson?.message) {
        message = errorJson.message
      }
    } catch {
      // Response body was not JSON; keep the generic status-based message.
    }
    throw new Error(message)
  }

  const json = (await response.json()) as {
    created?: number
    data: Array<{ b64_json?: string; url?: string }>
  }

  if (!json.data?.length) {
    throw new Error('Image generation response contains no image data')
  }

  const results: string[] = []
  for (const item of json.data) {
    if (typeof item.b64_json === 'string' && item.b64_json.length > 0) {
      const mediaType = detectImageMediaTypeFromBase64(item.b64_json)
      results.push(`data:${mediaType};base64,${item.b64_json}`)
    } else if (typeof item.url === 'string' && item.url.length > 0) {
      if (item.url.startsWith('data:')) {
        results.push(item.url)
      } else {
        // Remote URL: fetch and convert to a base64 data URL so downstream
        // blob storage can persist it the same way as inline images.
        const dataUrl = await convertImageUrlToBase64DataUrl(item.url, resolvedFetch)
        results.push(dataUrl)
      }
    } else {
      throw new Error('Image generation response missing image data')
    }
  }
  return results
}

function detectImageMediaTypeFromBase64(base64: string): string {
  // JPEG: starts with /9j/ (FFD8 in base64)
  if (base64.startsWith('/9j/')) return 'image/jpeg'
  // PNG: starts with iVBOR (89504E47 in base64)
  if (base64.startsWith('iVBOR')) return 'image/png'
  // GIF: starts with R0lGOD (474946 in base64)
  if (base64.startsWith('R0lGOD')) return 'image/gif'
  // WebP: starts with UklGR (52494646 in base64)
  if (base64.startsWith('UklGR')) return 'image/webp'
  return 'image/png'
}

export async function convertImageUrlToBase64DataUrl(url: string, fetchFn: typeof globalThis.fetch): Promise<string> {
  if (url.startsWith('data:')) return url
  const response = await fetchFn(url)
  if (!response.ok) {
    throw new Error(`Failed to fetch generated image (status ${response.status})`)
  }
  const blob = await response.blob()
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader()
    reader.onloadend = () => {
      if (typeof reader.result === 'string') resolve(reader.result)
      else reject(new Error('Failed to read generated image as data URL'))
    }
    reader.onerror = () => reject(new Error('Failed to read generated image as data URL'))
    reader.readAsDataURL(blob)
  })
}
