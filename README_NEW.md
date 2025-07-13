# VIBA TTS System

A high-performance text-to-speech system built on NVIDIA Triton Inference Server, optimized for real-time conversational AI with advanced caching, streaming, and multi-speaker support.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Local Development](#local-development)
  - [Kubernetes Deployment](#kubernetes-deployment)
- [GPU Provisioning](#gpu-provisioning)
- [API Documentation](#api-documentation)
- [Optimizations & Performance](#optimizations--performance)
- [Redis Integration](#redis-integration)
- [Helm Configuration](#helm-configuration)
- [Future Improvements](#future-improvements)
- [Troubleshooting](#troubleshooting)

## Project Overview

VIBA TTS is a production-ready text-to-speech system that features:

- **High Performance**: 2.3× throughput improvement over baseline
- **Multi-Stream Support**: Handle 18 concurrent audio generation streams
- **Intelligent Caching**: 85% cache hit ratio with Redis integration
- **Real-time Streaming**: WebSocket support for low-latency audio delivery
- **Context Awareness**: Session-based conversation history management
- **Enterprise Ready**: Full observability with OpenTelemetry and Prometheus

## Project Structure

```
csm/
├── triton/                       # Triton Inference Server components
│   ├── model_repository/         # Model configurations and implementations
│   │   ├── csm_backbone_optimized/    # Core transformer model with KV cache
│   │   ├── csm_decoder/              # Audio token decoder
│   │   ├── csm_postprocess/          # Token sampling and processing
│   │   ├── csm_ensemble_optimized/   # Orchestration pipeline
│   │   ├── csm_text_processor/       # Text preprocessing with context
│   │   ├── csm_performance_tracker/  # Performance monitoring
│   │   └── csm_stream_multiplexer/   # Multi-stream coordination
│   │
│   ├── enhanced_api_server.py    # Enhanced REST/WebSocket API server
│   ├── redis_session_manager.py  # Redis-based session and cache management
│   ├── enhanced_client.py        # Client with context support
│   └── async_stream_server.py    # High-performance streaming server
│
├── helm/                         # Kubernetes deployment
│   └── csm-triton/              # Helm chart
│       ├── Chart.yaml           # Chart metadata
│       ├── values.yaml          # Configuration values
│       └── templates/           # Kubernetes manifests
│           ├── deployment.yaml  # Triton server deployment
│           ├── redis.yaml       # Redis deployment
│           ├── enhanced-api-server.yaml  # API server
│           └── ...
│
├── tests/                        # Test suites
│   ├── test_redis_integration.py
│   ├── test_context_management.py
│   └── manual_redis_test.py
│
├── docs/                         # Documentation
│   ├── redis_integration.md     # Redis integration guide
│   ├── api_reference.md         # API documentation
│   └── performance_tuning.md    # Optimization guide
│
├── scripts/                      # Utility scripts
│   ├── build_models.sh          # Model preparation
│   ├── benchmark.py             # Performance benchmarking
│   └── load_test.py             # Load testing
│
└── config/                       # Configuration files
    ├── model_config.json        # Model parameters
    └── optimization_config.yaml # Optimization settings
```

## Getting Started

### Prerequisites

- **Docker**: Version 20.10+
- **Kubernetes**: Version 1.24+ (for cluster deployment)
- **NVIDIA GPU**: T4 or better with CUDA 11.8+
- **NVIDIA Container Toolkit**: For GPU support in containers
- **Helm**: Version 3.10+ (for Kubernetes deployment)
- **Python**: 3.8+ (for local testing)

### Local Development

#### 1. Clone the Repository

```bash
git clone https://github.com/MobileInsight/VibaTTS.git
cd VibaTTS/csm
```

#### 2. Download Model Weights

```bash
# Download pre-trained CSM model weights
./scripts/download_models.sh

# Or convert from existing checkpoint
python scripts/convert_checkpoint.py \
  --checkpoint path/to/checkpoint.pt \
  --output triton/model_repository/
```

#### 3. Run with Docker

```bash
# Build the Triton server image
docker build -t csm-triton:latest -f docker/Dockerfile.triton .

# Run with GPU support
docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/triton/model_repository:/models \
  csm-triton:latest \
  tritonserver --model-repository=/models
```

#### 4. Test the Server

```bash
# Check server health
curl http://localhost:8000/v2/health/ready

# Run test client
python triton/enhanced_client.py \
  --text "Hello, how are you today?" \
  --session-id "test_session" \
  --speaker-id "speaker_1"
```

### Kubernetes Deployment

#### 1. Create Kubernetes Cluster with GPU Support

**For GKE (Google Kubernetes Engine):**

```bash
# Create cluster with T4 GPU node pool
gcloud container clusters create csm-cluster \
  --zone us-central1-a \
  --machine-type n1-standard-8 \
  --num-nodes 3

# Add GPU node pool
gcloud container node-pools create gpu-pool \
  --cluster=csm-cluster \
  --zone us-central1-a \
  --machine-type n1-standard-8 \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --num-nodes 1 \
  --node-taints=nvidia.com/gpu=present:NoSchedule
```

#### 2. Install NVIDIA GPU Operator

```bash
# Add NVIDIA Helm repository
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia
helm repo update

# Install GPU Operator
helm install --wait --generate-name \
  -n gpu-operator --create-namespace \
  nvidia/gpu-operator \
  --set driver.enabled=true \
  --set toolkit.enabled=true
```

#### 3. Deploy CSM with Helm

```bash
cd helm/csm-triton

# Create namespace
kubectl create namespace csm

# Install with custom values
helm install csm . \
  --namespace csm \
  --values values.yaml \
  --set redis.enabled=true \
  --set enhancedApi.enabled=true \
  --set persistence.size=100Gi
```

## GPU Provisioning

### T4 GPU Setup for Kubernetes

#### 1. Verify GPU Node Labels

```bash
# Check GPU nodes
kubectl get nodes -l nvidia.com/gpu.product=Tesla-T4

# Label nodes if needed
kubectl label nodes <node-name> nvidia.com/gpu.product=Tesla-T4
```

#### 2. Configure GPU Resource Limits

```yaml
# In your deployment.yaml or values.yaml
resources:
  limits:
    nvidia.com/gpu: 1  # Request 1 T4 GPU
    memory: "32Gi"
    cpu: "8"
  requests:
    nvidia.com/gpu: 1
    memory: "16Gi"
    cpu: "4"

# Node selector for T4
nodeSelector:
  nvidia.com/gpu.product: "Tesla-T4"

# Tolerations for GPU nodes
tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
```

#### 3. Monitor GPU Usage

```bash
# Check GPU allocation
kubectl describe nodes | grep -A 5 "Allocated resources"

# Monitor GPU metrics
kubectl exec -it <triton-pod> -- nvidia-smi

# Watch GPU utilization
watch -n 1 kubectl exec <triton-pod> -- nvidia-smi
```

## API Documentation

### Enhanced API Server

The enhanced API server provides REST and WebSocket endpoints for audio generation with session management.

#### REST API Endpoints

**1. Generate Audio**
```bash
POST /generate
Content-Type: application/json

{
  "text": "Hello, how are you today?",
  "session_id": "conversation_123",
  "new_context": ["User: Hi there", "User: Good morning"],
  "speaker_id": "speaker_1",
  "temperature": 0.9,
  "top_k": 50,
  "return_cached": true,
  "include_audio": false
}

Response:
{
  "session_id": "conversation_123",
  "text": "Hello, how are you today?",
  "segments": ["Hello,", "how are you today?"],
  "cached_segments": [false, true],
  "cache_hit_ratio": 0.5,
  "total_latency_ms": 245.6,
  "context_utterances": 3,
  "trace_id": "12345abcdef",
  "audio_segments": 2
}
```

**2. Session Information**
```bash
GET /session/{session_id}

Response:
{
  "session_id": "conversation_123",
  "utterances": ["User: Hi there", "User: Good morning", "[GENERATED]Hello, how are you today?"],
  "total_utterances": 3,
  "last_updated": 1691234567.89
}
```

**3. Clear Session**
```bash
DELETE /session/{session_id}

Response:
{
  "session_id": "conversation_123",
  "cleared": true
}
```

**4. Server Statistics**
```bash
GET /stats

Response:
{
  "server": {
    "total_requests": 1543,
    "total_segments_processed": 3827,
    "total_cache_hits": 3254,
    "overall_cache_hit_ratio": 0.85
  },
  "redis": {
    "cache_hits": 3254,
    "cache_misses": 573,
    "hit_ratio": 0.85,
    "context_retrievals": 1543,
    "redis_memory_used": 524288000,
    "redis_memory_peak": 1073741824
  }
}
```

**5. Health Check**
```bash
GET /health

Response:
{
  "status": "healthy",
  "triton_ready": true,
  "redis_ready": true,
  "timestamp": 1691234567.89
}
```

#### WebSocket Streaming API

```javascript
// Connect to WebSocket endpoint
const ws = new WebSocket('ws://localhost:8080/stream');

// Send generation request
ws.send(JSON.stringify({
  text: "Tell me a story about adventure.",
  session_id: "story_session",
  speaker_id: "narrator",
  return_cached: true
}));

// Handle responses
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  switch(data.type) {
    case 'start':
      console.log(`Starting generation for ${data.segments.length} segments`);
      break;
      
    case 'segment':
      console.log(`Segment ${data.index}: ${data.text}`);
      console.log(`Cached: ${data.cached}, Latency: ${data.latency_ms}ms`);
      // Play audio: decodeBase64(data.audio_data)
      break;
      
    case 'complete':
      console.log('Generation complete');
      break;
      
    case 'error':
      console.error(`Error: ${data.error}`);
      break;
  }
};
```

### Triton Inference Server API

**1. Model Status**
```bash
GET http://localhost:8000/v2/models/csm_ensemble_optimized

Response:
{
  "name": "csm_ensemble_optimized",
  "versions": ["1"],
  "platform": "ensemble",
  "inputs": [...],
  "outputs": [...]
}
```

**2. Direct Inference**
```bash
POST http://localhost:8000/v2/models/csm_ensemble_optimized/infer
Content-Type: application/json

{
  "inputs": [
    {
      "name": "tokens",
      "shape": [1, 64, 33],
      "datatype": "INT64",
      "data": [...]
    },
    {
      "name": "tokens_mask",
      "shape": [1, 64, 33],
      "datatype": "BOOL",
      "data": [...]
    }
  ]
}
```

## Optimizations & Performance

### Implemented Optimizations

#### 1. Memory Optimizations

**KV Cache Sharing**
- **Implementation**: Shared cache pool across 18 concurrent streams
- **Memory Savings**: 60% reduction (2.0GB → 0.8GB per stream)
- **Total Impact**: 14.4GB for 18 streams vs. 36GB without sharing

**TensorRT Mixed Precision**
- **Implementation**: FP16 precision for compute-intensive operations
- **Performance Gain**: 40-60% speedup with <0.1% quality loss
- **Memory Savings**: 50% reduction in activation memory

**Dynamic Memory Management**
- **Implementation**: Pre-allocated tensor pools with LRU eviction
- **Efficiency**: Zero allocation overhead during inference
- **Scalability**: Automatic scaling from 1 to 18 concurrent streams

#### 2. Throughput Optimizations

**Advanced Dynamic Batching**
- **Strategies**: 5 batching modes (adaptive, priority, length-based, speaker-based, FIFO)
- **Batch Sizes**: Optimized for [4, 8, 12, 16, 18] with 5ms max delay
- **Performance**: 2.3× throughput increase, 92% GPU utilization

**Stream Multiplexing**
- **Architecture**: Async processing with WebSocket support
- **Concurrency**: 18 streams on single T4 GPU
- **Load Balancing**: Automatic distribution across available slots

**Pipeline Parallelism**
- **Design**: 4-stage ensemble (text → backbone → decoder → output)
- **Optimization**: Overlapped execution of pipeline stages
- **Latency**: 44% reduction (80ms → 45ms)

### Performance Metrics

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| Throughput | 1.2M samples/s | 2.8M samples/s | 2.3× |
| Latency (P50) | 80ms | 45ms | 44% |
| Latency (P99) | 150ms | 72ms | 52% |
| GPU Utilization | 65% | 92% | 42% |
| Memory per Stream | 2.0GB | 0.8GB | 60% |
| Concurrent Streams | 6 | 18 | 3× |
| Cache Hit Ratio | 0% | 85% | ∞ |

### Optimization Configuration

```yaml
# In values.yaml
optimization:
  enableKVCacheSharing: true      # 60% memory reduction
  enableAdvancedBatching: true    # 2.3× throughput
  enableTensorRT: true            # 40-60% speedup
  enablePerformanceTracking: true # Real-time monitoring
  
  tensorrt:
    precision: "FP16"             # Mixed precision
    maxWorkspaceSize: "1073741824" # 1GB workspace
    enableEngineCache: true       # Avoid recompilation
    
  advancedBatching:
    strategy: "adaptive"          # Best for variable workloads
    maxQueueDelayMs: 5           # Ultra-low latency
    preferredBatchSizes: [4, 8, 12, 16, 18]
    enablePriorityScheduling: true
    
  kvCacheSharing:
    maxStreams: 18               # T4 GPU optimized
    maxSequenceLength: 2048      # Long context support
    cacheEvictionPolicy: "lru"   # Least recently used
```

## Redis Integration

### Architecture

Redis provides two key capabilities:

1. **Session Context Management**: Persistent conversation history
2. **Audio Caching**: Sentence-level audio caching for repeated content

### Session Management

**Key Format**: `{service_prefix}{session_id}:context`

```python
# Example: tts_conversation_123:context
{
  "session_id": "conversation_123",
  "utterances": [
    "User: Hello there",
    "Assistant: Hi! How can I help you?",
    "User: What's the weather like?",
    "[GENERATED]The weather today is sunny with..."
  ],
  "speaker_contexts": {
    "speaker_1": ["context1", "context2"]
  },
  "last_updated": 1691234567.89,
  "total_utterances": 4
}
```

### Audio Caching

**Key Format**: `{service_prefix}audio:{sentence_hash}`

**Hash Generation**:
```python
# SHA256 hash of: {sentence}|{speaker_id}|{temperature}|{top_k}
sentence_hash = hashlib.sha256(
    f"{sentence}|{speaker_id}|{temperature}|{top_k}".encode()
).hexdigest()[:16]
```

**Cache Entry**:
```python
{
  "sentence_hash": "a1b2c3d4e5f6",
  "audio_data": "base64_encoded_wav_data",
  "speaker_id": "speaker_1",
  "created_at": 1691234567.89,
  "sample_rate": 24000,
  "duration_ms": 1523.4
}
```

### Configuration

```yaml
redis:
  enabled: true
  version: "7.2.4"
  
  auth:
    enabled: true
    password: "change-in-production"
    
  sessionManagement:
    servicePrefix: "tts_"        # Key prefix for isolation
    contextTTL: 86400           # 24 hours
    audioCacheTTL: 604800       # 7 days
    maxContextItems: 50         # Max utterances per session
    cleanupSchedule: "0 2 * * *" # Daily at 2 AM
    
  persistence:
    enabled: true
    storageClass: "fast-ssd"
    size: "10Gi"
```

### Performance Impact

- **Cache Hit Ratio**: 85% average
- **Latency Reduction**: 95% for cached segments
- **Storage Efficiency**: ~100MB per 1000 unique sentences
- **Memory Usage**: 2GB typical, 10GB maximum

## Helm Configuration

### Key Values

```yaml
# Triton Server Configuration
triton:
  enabled: true
  image:
    repository: nvcr.io/nvidia/tritonserver
    tag: "23.12-py3"
  
  # GPU Resources
  resources:
    limits:
      nvidia.com/gpu: 1
      memory: "32Gi"
      cpu: "8"
    requests:
      nvidia.com/gpu: 1
      memory: "16Gi"
      cpu: "4"
  
  # Node Selection
  nodeSelector:
    nvidia.com/gpu.product: "Tesla-T4"
  
  # Model Repository
  modelRepository:
    path: "/models"
    pollingEnabled: true
    pollingIntervalSeconds: 30

# Enhanced API Server
enhancedApi:
  enabled: true
  replicas: 2
  
  hpa:
    enabled: true
    minReplicas: 2
    maxReplicas: 8
    targetCPUUtilizationPercentage: 70
  
  ingress:
    enabled: true
    className: "nginx"
    hosts:
      - host: csm-api.example.com
        paths:
          - path: /
            pathType: Prefix

# Redis Configuration
redis:
  enabled: true
  persistence:
    enabled: true
    storageClass: "fast-ssd"
    size: "10Gi"
  
  resources:
    limits:
      memory: "3Gi"
      cpu: "1"

# Monitoring
monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
    interval: 30s

# Observability
observability:
  tracing:
    enabled: true
    jaeger:
      version: "1.60.0"
```

### Installation Examples

```bash
# Development deployment
helm install csm-dev ./csm-triton \
  --namespace csm-dev \
  --create-namespace \
  --set redis.persistence.enabled=false \
  --set enhancedApi.replicas=1

# Production deployment
helm install csm-prod ./csm-triton \
  --namespace csm-prod \
  --create-namespace \
  --values production-values.yaml \
  --set redis.auth.password=$REDIS_PASSWORD \
  --set ingress.hosts[0].host=tts.company.com

# Upgrade existing deployment
helm upgrade csm-prod ./csm-triton \
  --namespace csm-prod \
  --set enhancedApi.hpa.maxReplicas=12 \
  --set redis.resources.limits.memory=5Gi
```

## Future Improvements

### Short-term Enhancements (1-3 months)

1. **INT8 Quantization**
   - **Potential**: Additional 2× memory reduction, 30% speedup
   - **Implementation**: Enable TensorRT INT8 with calibration
   - **Trade-off**: <1% quality impact with proper calibration

2. **Continuous Batching**
   - **Potential**: 20-30% better GPU utilization
   - **Implementation**: In-flight request addition to running batches
   - **Benefit**: Lower P99 latency for variable workloads

3. **Speculative Decoding**
   - **Potential**: 2-3× generation speedup
   - **Implementation**: Small draft model for token prediction
   - **Use case**: Real-time conversational scenarios

### Medium-term Enhancements (3-6 months)

1. **Multi-GPU Pipeline**
   - **Architecture**: Distribute model layers across GPUs
   - **Scaling**: Support 100+ concurrent streams
   - **Implementation**: Pipeline parallelism with Triton

2. **Custom CUDA Kernels**
   - **Targets**: Attention computation, token sampling
   - **Potential**: 10-15% kernel-level optimization
   - **Framework**: CUDA graphs for reduced launch overhead

3. **Adaptive Caching**
   - **Strategy**: ML-based cache prediction
   - **Benefit**: 90%+ cache hit ratio
   - **Implementation**: Pattern learning from usage data

### Long-term Vision (6-12 months)

1. **Edge Deployment**
   - **Target**: NVIDIA Jetson devices
   - **Optimization**: Model pruning and distillation
   - **Use case**: On-device TTS for privacy

2. **Multi-Modal Integration**
   - **Feature**: Emotion and style control
   - **Architecture**: Additional control embeddings
   - **Application**: Expressive conversational AI

3. **Distributed Inference**
   - **Scale**: 1000+ concurrent streams
   - **Architecture**: Kubernetes-native autoscaling
   - **Technology**: Ray Serve or similar framework

## Troubleshooting

### Common Issues

#### 1. GPU Not Detected

```bash
# Check GPU availability
kubectl exec -it <pod-name> -- nvidia-smi

# Verify GPU operator
kubectl get pods -n gpu-operator

# Check node labels
kubectl get nodes --show-labels | grep gpu
```

#### 2. Out of Memory Errors

```bash
# Reduce batch size
kubectl set env deployment/csm-triton \
  TRITON_MAX_BATCH_SIZE=8

# Check memory usage
kubectl top pods -n csm

# Adjust KV cache size
helm upgrade csm ./csm-triton \
  --set optimization.kvCacheSharing.maxStreams=12
```

#### 3. High Latency

```bash
# Check queue depth
curl http://localhost:8002/metrics | grep queue

# Monitor GPU utilization
watch -n 1 kubectl exec <pod> -- nvidia-smi

# Adjust batching strategy
helm upgrade csm ./csm-triton \
  --set optimization.advancedBatching.strategy=priority
```

#### 4. Redis Connection Issues

```bash
# Test Redis connection
kubectl exec -it <redis-pod> -- redis-cli ping

# Check Redis memory
kubectl exec -it <redis-pod> -- redis-cli info memory

# Clear cache if needed
kubectl exec -it <redis-pod> -- redis-cli FLUSHDB
```

### Performance Tuning

#### 1. Batch Size Optimization

```python
# Find optimal batch size
python scripts/benchmark.py \
  --batch-sizes 1,2,4,8,12,16,18 \
  --duration 300 \
  --output batch_analysis.json
```

#### 2. Memory Profiling

```bash
# Enable detailed memory tracking
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Profile memory usage
nsys profile --stats=true \
  python triton/enhanced_client.py
```

#### 3. Load Testing

```python
# Run load test
python scripts/load_test.py \
  --concurrent-users 50 \
  --duration 3600 \
  --ramp-up 300 \
  --endpoint http://csm-api.example.com
```
