import type { ModelDependencies } from 'src/shared/types/adapters'
import type { SentryScope } from 'src/shared/utils/sentry_adapter'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { generateImageWithOpenAICompatibleAPI, getOpenAIImageGenerationOptions } from './openai-compatible-image'
import OpenRouter from './openrouter'

const aiMocks = vi.hoisted(() => ({
  generateImage: vi.fn(),
}))

vi.mock('ai', async (importOriginal) => {
  const actual = await importOriginal<typeof import('ai')>()
  return {
    ...actual,
    generateImage: aiMocks.generateImage,
  }
})

const openRouterMocks = vi.hoisted(() => {
  const imageModel = { modelId: 'openrouter-image-model' }
  const imageModelFactory = vi.fn(() => imageModel)
  const languageModel = vi.fn()
  const createOpenRouter = vi.fn(
    (_options: { apiKey: string; headers: Record<string, string>; fetch: typeof globalThis.fetch }) => ({
      imageModel: imageModelFactory,
      languageModel,
    })
  )
  return { createOpenRouter, imageModel, imageModelFactory }
})

vi.mock('@openrouter/ai-sdk-provider', () => ({
  createOpenRouter: openRouterMocks.createOpenRouter,
}))

class TestOpenRouter extends OpenRouter {
  public exposeImageModel() {
    return this.getImageModel()
  }
}

function createDependencies(overrides: Partial<ModelDependencies> = {}): ModelDependencies {
  return {
    request: {
      apiRequest: vi.fn(),
      fetchWithOptions: vi.fn(),
    },
    storage: {
      saveImage: vi.fn(),
      getImage: vi.fn(),
    },
    sentry: {
      withScope: vi.fn((callback: (scope: SentryScope) => void) => callback({ setTag: vi.fn(), setExtra: vi.fn() })),
      captureException: vi.fn(),
    },
    getRemoteConfig: vi.fn().mockReturnValue({ setting_chatboxai_first: false }),
    ...overrides,
  }
}

function jsonResponse(body: unknown, init: { status?: number; headers?: Record<string, string> } = {}): Response {
  return new Response(JSON.stringify(body), {
    status: init.status ?? 200,
    headers: { 'Content-Type': 'application/json', ...(init.headers ?? {}) },
  })
}

describe('OpenAI-compatible image models', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('maps Image Creator aspect ratios to OpenAI image sizes', () => {
    expect(getOpenAIImageGenerationOptions('gpt-image-1', 'auto')).toEqual({})
    expect(getOpenAIImageGenerationOptions('gpt-image-1', '1:1')).toEqual({ size: '1024x1024' })
    expect(getOpenAIImageGenerationOptions('gpt-image-1', '3:2')).toEqual({ size: '1536x1024' })
    expect(getOpenAIImageGenerationOptions('gpt-image-1', '2:3')).toEqual({ size: '1024x1536' })
    expect(getOpenAIImageGenerationOptions('dall-e-3', '7:4')).toEqual({ size: '1792x1024' })
    expect(getOpenAIImageGenerationOptions('dall-e-3', '4:7')).toEqual({ size: '1024x1792' })
    expect(getOpenAIImageGenerationOptions('dall-e-2', '3:2')).toEqual({})
  })

  describe('generateImageWithOpenAICompatibleAPI', () => {
    it('parses b64_json responses and tags the media type from magic bytes', async () => {
      const fetch = vi.fn().mockResolvedValue(jsonResponse({ created: 1, data: [{ b64_json: '/9j/4AAQSkZJRgABAQ' }] }))
      const result = await generateImageWithOpenAICompatibleAPI({
        apiKey: 'k',
        apiHost: 'https://example.test/v1',
        modelId: 'gpt-image-2',
        prompt: 'a red apple',
        n: 1,
        fetch,
      })
      expect(fetch).toHaveBeenCalledWith(
        'https://example.test/v1/images/generations',
        expect.objectContaining({ method: 'POST' })
      )
      expect(result).toEqual(['data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ'])
    })

    it('parses url data URLs without re-fetching', async () => {
      const fetch = vi.fn().mockResolvedValue(jsonResponse({ data: [{ url: 'data:image/png;base64,iVBORw0KGgo=' }] }))
      const result = await generateImageWithOpenAICompatibleAPI({
        apiKey: 'k',
        apiHost: 'https://example.test/v1',
        modelId: 'm',
        prompt: 'p',
        n: 1,
        fetch,
      })
      expect(result).toEqual(['data:image/png;base64,iVBORw0KGgo='])
      // Should not attempt a second fetch to resolve the data URL.
      expect(fetch).toHaveBeenCalledOnce()
    })

    it('extracts the server error message on failure', async () => {
      const fetch = vi
        .fn()
        .mockResolvedValue(
          jsonResponse(
            { error: { message: '分组 Free 下模型 gpt-image-2 的可用渠道不存在', type: 'new_api_error' } },
            { status: 500 }
          )
        )
      await expect(
        generateImageWithOpenAICompatibleAPI({
          apiKey: 'k',
          apiHost: 'https://example.test/v1',
          modelId: 'gpt-image-2',
          prompt: 'p',
          n: 1,
          fetch,
        })
      ).rejects.toThrow('分组 Free 下模型 gpt-image-2 的可用渠道不存在')
    })

    it('falls back to a status-based message when the error body is not JSON', async () => {
      const fetch = vi.fn().mockResolvedValue(new Response('Gateway Error', { status: 502 }))
      await expect(
        generateImageWithOpenAICompatibleAPI({
          apiKey: 'k',
          apiHost: 'https://example.test/v1',
          modelId: 'm',
          prompt: 'p',
          n: 1,
          fetch,
        })
      ).rejects.toThrow('Image generation failed with status 502')
    })

    it('throws when the response contains no image data', async () => {
      const fetch = vi.fn().mockResolvedValue(jsonResponse({ data: [] }))
      await expect(
        generateImageWithOpenAICompatibleAPI({
          apiKey: 'k',
          apiHost: 'https://example.test/v1',
          modelId: 'm',
          prompt: 'p',
          n: 1,
          fetch,
        })
      ).rejects.toThrow('Image generation response contains no image data')
    })
  })

  it('creates an image model through the OpenRouter provider', () => {
    const dependencies = createDependencies()
    const instance = new TestOpenRouter(
      {
        apiKey: 'or-key',
        model: { modelId: 'google/gemini-3-pro-image', type: 'image' },
      },
      dependencies
    )

    expect(instance.exposeImageModel()).toBe(openRouterMocks.imageModel)
    expect(openRouterMocks.createOpenRouter).toHaveBeenCalledWith({
      apiKey: 'or-key',
      headers: {
        'HTTP-Referer': 'https://chatboxai.app',
        'X-Title': 'Chatbox AI',
      },
      fetch: expect.any(Function),
    })
    expect(openRouterMocks.imageModelFactory).toHaveBeenCalledWith('google/gemini-3-pro-image')
  })

  it('passes OpenRouter reference images through the AI SDK image prompt', async () => {
    aiMocks.generateImage.mockResolvedValue({ images: [] })
    const instance = new TestOpenRouter(
      {
        apiKey: 'or-key',
        model: { modelId: 'google/gemini-3-pro-image', type: 'image' },
      },
      createDependencies()
    )

    await instance.paint({
      prompt: 'edit this image',
      images: [{ imageUrl: 'https://example.test/reference.png' }, { imageUrl: 'data:image/png;base64,AQID' }],
      num: 1,
      aspectRatio: '16:9',
    })

    expect(aiMocks.generateImage).toHaveBeenCalledWith(
      expect.objectContaining({
        prompt: {
          text: 'edit this image',
          images: ['https://example.test/reference.png', 'data:image/png;base64,AQID'],
        },
        n: 1,
        aspectRatio: '16:9',
        maxRetries: 0,
      })
    )
  })

  it('persists each OpenRouter image before starting the next request', async () => {
    const instance = new TestOpenRouter(
      {
        apiKey: 'or-key',
        model: { modelId: 'google/gemini-3-pro-image', type: 'image' },
      },
      createDependencies()
    )
    const callback = vi.fn()
    const secondError = new Error('second image failed')
    aiMocks.generateImage
      .mockResolvedValueOnce({ images: [{ mediaType: 'image/png', base64: 'first-image' }] })
      .mockRejectedValueOnce(secondError)

    await expect(
      instance.paint({ prompt: 'two images', num: 2, aspectRatio: '16:9' }, undefined, callback)
    ).rejects.toBe(secondError)

    expect(aiMocks.generateImage).toHaveBeenCalledTimes(2)
    expect(aiMocks.generateImage).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({ n: 1, aspectRatio: '16:9', maxRetries: 0 })
    )
    expect(aiMocks.generateImage).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({ n: 1, aspectRatio: '16:9', maxRetries: 0 })
    )
    expect(callback).toHaveBeenCalledOnce()
    expect(callback).toHaveBeenCalledWith('data:image/png;base64,first-image')
  })
})
