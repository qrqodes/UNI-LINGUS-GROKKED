# Setting Up Open-Source LLM Models for Translation

This guide explains how to set up and configure open-source language models (DeepSeek, Llama 4, Mixtral) to enhance the translation capabilities of your bot.

## Benefits of Using Open-Source Models

- **Cost savings**: Reduce API costs by using locally hosted models instead of OpenAI/Anthropic
- **Privacy**: Keep data on your own servers without sending it to third-party APIs
- **Offline capability**: Work without internet connection to cloud services
- **Customization**: Fine-tune models for specific language pairs or domains

## Requirements

- A server with sufficient RAM and GPU (recommended)
  - DeepSeek: 16GB+ RAM, CUDA-compatible GPU with 8GB+ VRAM recommended
  - Llama 4: 16GB+ RAM, CUDA-compatible GPU with 8GB+ VRAM recommended
  - Mixtral: 12GB+ RAM, CUDA-compatible GPU with 6GB+ VRAM recommended
- Docker (for containerized deployment)
- Basic knowledge of terminal commands

## Setup Instructions

### 1. Install Model Servers

You can use any one (or all) of these models. Each one needs to be configured separately.

#### Option A: Set up with Hugging Face Text Generation Interface (TGI)

```bash
# Clone the TGI repository
git clone https://github.com/huggingface/text-generation-inference
cd text-generation-inference

# Download and run DeepSeek model
docker run --gpus all --shm-size 1g -p 8080:80 \
    -v $HOME/.cache/huggingface:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id deepseek-ai/deepseek-coder-6.7b-instruct

# For Llama 4 (requires accepting Meta's license on Hugging Face)
docker run --gpus all --shm-size 1g -p 8081:80 \
    -v $HOME/.cache/huggingface:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Meta-Llama-3-8B-Instruct

# For Mixtral
docker run --gpus all --shm-size 1g -p 8082:80 \
    -v $HOME/.cache/huggingface:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id mistralai/Mixtral-8x7B-Instruct-v0.1
```

#### Option B: Use LocalAI (easier for CPU-only setup)

```bash
# Clone LocalAI
git clone https://github.com/go-skynet/LocalAI
cd LocalAI

# Start with docker-compose
docker-compose up -d

# Download models through the web UI at http://localhost:8080
# Or via API
curl http://localhost:8080/models/apply -H "Content-Type: application/json" \
  -d '{"url": "https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-instruct/resolve/main/model.gguf"}'
```

### 2. Set Environment Variables

Add these environment variables to your application with the URLs of your model servers. Only set the ones for models you've deployed:

```
DEEPSEEK_SERVER_URL=http://your-server-ip:8080
LLAMA_SERVER_URL=http://your-server-ip:8081
MIXTRAL_SERVER_URL=http://your-server-ip:8082
```

### 3. Testing Model Connectivity

Use these curl commands to test if your models are working correctly:

```bash
# Test DeepSeek
curl -X POST http://your-server-ip:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Translate this to Spanish: Hello world", "max_tokens": 100, "temperature": 0.7}'

# Test Llama
curl -X POST http://your-server-ip:8081/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Translate this to French: Hello world", "max_tokens": 100, "temperature": 0.7}'
```

## Further Customization

### Fine-tuning for Translation

For even better translation results, you can fine-tune these models on translation datasets:

1. Collect parallel text datasets for your language pairs (e.g., from OPUS: http://opus.nlpl.eu/)
2. Format the data for instruction fine-tuning
3. Use Hugging Face's fine-tuning scripts to create a specialized translation model

### Quantized Models for Lower Resource Requirements

If you have limited resources, consider using quantized models which require less memory:

- 4-bit quantized models can run on consumer hardware with 8GB RAM
- GGUF format models work well with llama.cpp for CPU-only deployment

## Troubleshooting

- **Out of memory errors**: Reduce model size or increase quantization
- **Slow responses**: Enable FP16 inference or use a smaller model
- **Connection errors**: Check firewall settings and ensure ports are open
- **Bad translations**: Try adjusting temperature (lower for more consistent results)

## Resources

- DeepSeek: https://github.com/deepseek-ai/DeepSeek-Coder
- Llama 4: https://ai.meta.com/llama/
- Mixtral: https://mistral.ai/news/mixtral-of-experts/
- Text Generation Inference: https://github.com/huggingface/text-generation-inference
- LocalAI: https://github.com/go-skynet/LocalAI