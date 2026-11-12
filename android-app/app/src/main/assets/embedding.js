import {env, pipeline} from './vendor/transformers.min.js'

env.allowRemoteModels = false
env.allowLocalModels = true
env.localModelPath = '/models/'
env.backends.onnx.wasm.wasmPaths = '/vendor/'
env.backends.onnx.wasm.numThreads = 1

const models = {
  e5: {name: 'Xenova/multilingual-e5-small', dtype: 'int8'},
  minilm: {name: 'sentence-transformers/all-MiniLM-L6-v2', dtype: 'fp32'}
}
const loaded = new Map()
let queue = Promise.resolve()

window.embedRequest = (id, model, textsJson, kind) => {
  queue = queue.catch(() => {}).then(async () => {
    try {
      const config = models[model]
      if (!config) throw new Error('Unknown embedding model')
      if (!loaded.has(model)) loaded.set(model, pipeline('feature-extraction', config.name, {dtype: config.dtype, device: 'wasm'}))
      const extractor = await loaded.get(model)
      const vectors = []
      for (const text of JSON.parse(textsJson)) {
        const input = model === 'e5' ? `${kind === 'query' ? 'query' : 'passage'}: ${text}` : text
        const output = await extractor(input, {pooling: 'mean', normalize: true})
        const vector = Array.from(output.data)
        if (vector.length !== 384 || vector.some(value => !Number.isFinite(value))) throw new Error('Invalid embedding output')
        vectors.push(vector)
      }
      EmbeddingCallbacks.complete(id, JSON.stringify(vectors))
    } catch (error) {
      loaded.delete(model)
      EmbeddingCallbacks.complete(id, JSON.stringify({error: error.message}))
    }
  })
}

EmbeddingCallbacks.ready()
