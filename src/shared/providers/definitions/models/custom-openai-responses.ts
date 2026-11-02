import { createOpenAI } from '@ai-sdk/openai'
import { extractReasoningMiddleware, wrapLanguageModel } from 'ai'
import AbstractAISDKModel from '../../../models/abstract-ai-sdk'
import { fetchRemoteModels } from '../../../models/openai-compatible'
import type { CallChatCompletionOptions } from '../../../models/types'
import { createFetchWithProxy } from '../../../models/utils/fetch-proxy'
import type { ProviderModelInfo } from '../../../types'
import type { ModelDependencies } from '../../../types/adapters'
import { normalizeOpenAIResponsesHostAndPath } from '../../../utils/llm_utils'
import { normalizeOpenAIReasoningOptions } from '../../../utils/reasoning-control'
import {
  compatibleImageApiHeaders,
  generateImageWithOpenAICompatibleAPI,
  getOpenAIImageGenerationOptions,
} from './openai-compatible-image'

interface Options {
  apiKey: string
  apiHost: string
  apiPath: string
  model: ProviderModelInfo
  temperature?: number
  topP?: number
  maxOutputTokens?: number
  stream?: boolean
  useProxy?: boolean
}

type FetchFunction = typeof globalThis.fetch

export default class CustomOpenAIResponses extends AbstractAISDKModel {
  public name = 'Custom OpenAI Responses'

  constructor(
    public options: Options,
    dependencies: ModelDependencies
  ) {
    super(options, dependencies)
    const { apiHost, apiPath } = normalizeOpenAIResponsesHostAndPath(options)
    this.options = { ...options, apiHost, apiPath }
  }

  protected getCallSettings(options: CallChatCompletionOptions) {
    return {
      temperature: this.options.temperature,
      topP: this.options.topP,
      maxOutputTokens: this.options.maxOutputTokens,
      stream: this.options.stream,
      providerOptions: {
        openai: {
          ...normalizeOpenAIReasoningOptions(this.options.model.modelId, options.providerOptions?.openai),
          store: false,
        },
      },
    }
  }

  static isSupportTextEmbedding() {
    return true
  }

  protected getProvider(_options: CallChatCompletionOptions, fetchFunction?: FetchFunction) {
    return createOpenAI({
      apiKey: this.options.apiKey,
      baseURL: this.options.apiHost,
      fetch: fetchFunction,
      headers: this.options.apiHost.includes('openrouter.ai')
        ? {
            'HTTP-Referer': 'https://chatboxai.app',
            'X-Title': 'Chatbox AI',
          }
        : this.options.apiHost.includes('aihubmix.com')
          ? {
              'APP-Code': 'VAFU9221',
            }
          : undefined,
    })
  }

  protected getChatModel(options: CallChatCompletionOptions) {
    const { apiHost, apiPath } = this.options
    const provider = this.getProvider(options, (_input, init) =>
      createFetchWithProxy(this.options.useProxy, this.dependencies)(`${apiHost}${apiPath}`, init)
    )
    return wrapLanguageModel({
      model: provider.responses(this.options.model.modelId),
      middleware: extractReasoningMiddleware({ tagName: 'think' }),
    })
  }

  public listModels() {
    return fetchRemoteModels(
      {
        apiHost: this.options.apiHost,
        apiKey: this.options.apiKey,
        useProxy: this.options.useProxy,
      },
      this.dependencies
    )
  }

  public async paint(
    params: {
      prompt: string
      images?: { imageUrl: string }[]
      num: number
      aspectRatio?: string
    },
    signal?: AbortSignal,
    callback?: (picBase64: string) => void | Promise<void>
  ): Promise<string[]> {
    const modelId = this.options.model.modelId
    if (!modelId) {
      throw new Error('Model ID is required for image generation')
    }

    const size = Object.entries(this.getImageGenerationOptions(params.aspectRatio)).find(([key]) => key === 'size')?.[1]

    const dataUrls: string[] = []
    for (let i = 0; i < params.num; i++) {
      const urls = await generateImageWithOpenAICompatibleAPI({
        apiKey: this.options.apiKey,
        apiHost: this.options.apiHost,
        modelId,
        prompt: params.prompt,
        n: 1,
        size,
        images: params.images,
        signal,
        fetch: createFetchWithProxy(this.options.useProxy, this.dependencies),
        headers: compatibleImageApiHeaders(this.options.apiHost),
      })
      for (const url of urls) {
        await callback?.(url)
        dataUrls.push(url)
      }
    }
    return dataUrls
  }

  protected override getImageGenerationOptions(aspectRatio?: string) {
    return getOpenAIImageGenerationOptions(this.options.model.modelId, aspectRatio)
  }

  protected override getMaxImagesPerGenerationRequest() {
    return 1
  }
}
