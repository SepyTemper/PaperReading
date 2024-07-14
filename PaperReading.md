# Paper Reading

## vLLM

### Background

现有的llm系统在生成新token时产生的 kv cache 巨大且动态增长和缩减，导致系统

- 产生内部碎片和外部碎片

- 无法进行内存共享

### Paged Attention

将请求的 kv cache 划分为块，每个块可以包含固定数量的令牌的注意力键和值。在 Paged Attention 中，可以像OS的虚拟内存一样更灵活地管理kv cache：可以将block视为page，将token视为byte，将request视为process。进而

- 使用相对较小的块并按需分配来减少内部内存划分

- 由于所有块大小相同，它消除外部内存划分

- 使内存能在block的粒度上，跨不同序列的同一请求甚至跨不同请求共享

### vLLM Keypoint

- kv cache manager：在PagedAttention的基础上，将kv cache组织为固定大小的kv block，类似于虚拟内存中的页面。一次请求的kv cache表示为一系列逻辑kv block，随着新token及其kv cache的生成，这些块从左到右逐步填充，最后一个kv block的未填充位置保留用于未来的生成。在 GPU 工作线程上，块引擎分配一块连续的 GPU DRAM，并将其划分为物理kv block。kv block管理器还维护块表——每个request的逻辑和物理kv block之间的映射。每个块表条目记录逻辑块对应的物理block和已填充位置的数量。将逻辑和物理kv block分开，允许 vLLM 动态增长kv cache内存，而无需提前为所有位置预留，从而消除了现有系统中的大部分内存浪费。

- 调度和抢占：当请求流量超过系统容量时，vLLM必须优先处理一部分请求。在vLLM中，采用先到先服务（FCFS）调度策略处理所有请求，确保公平并防止starvation的产生。当vLLM需要抢占请求时，它确保最早到达的请求优先处理，最新的请求最先被抢占。

- 分布式执行：GPU 工作器不需要在内存管理上进行同步，它们只需要在每个解码迭代的开始时接收所有内存管理信息以及步骤输入。

    vLLM在处理参数规模过大的情况时使用 Megatron-LM 风格的tensor模型并行策略，这种策略遵循了SPMD的执行模式，其中线性层被分区以进行块状矩阵乘法，并且GPU通过全局归约操作不断同步中间结果。即使使用模型并行执行，每个模型分片仍然处理相同的输入标记集，因此需要相同位置的kv cache。因此，vLLM在集中式调度器内具有单个kv cache管理器。不同的GPU工作器共享该管理器，以及从逻辑块到物理块的映射。这种共享的映射允许GPU工作器使用调度器为每个输入请求提供的物理块执行模型。虽然每个GPU工作器具有相同的物理块ID，但工作器只存储其对应注意力头的kv cache的一部分。在每一步中，调度器首先为批处理中的每个请求准备包含输入标记 ID 和每个请求的块表的控制消息。然后，调度器将这个控制消息广播给 GPU 工作器。接着，GPU 工作器开始使用输入标记 ID 执行模型。在注意力层中，GPU 工作器根据控制消息中的块表读取kv cache。在执行过程中，GPU 工作器使用全局归约通信原语同步中间结果，无需调度器协调。最后，GPU 工作器将本次迭代中抽样的标记发送回调度器。

## DeepspeedInference

### Background

在生产环境中使用基于Transformer的模型的在线场景需要满足严格的延迟要求，因此使用的批处理大小通常较小。而小批量的性能受限于读取模型权重时的内存带宽利用率，所以需要优化小批量的内存带宽利用率，这面临着三个主要挑战:

- 由于在执行小批量Transformer层操作时，不同内核的工作量有限，推理性能受到内核调用开销的影响。

- 每次内核调用都会将数据写入全局内存，而这些数据在下一次内核调用时被GPU核心读取，这种GPU核心与全局内存之间的数据传输增加了额外的开销。

- cuBLAS和CUTLASS GeMM库对于极小批量的优化不佳，无法实现良好的内存带宽利用率。

大批量推理性能则受到计算资源利用率的限制。在Transformer层内部，像GeMM这样的计算密集型操作可以通过使用CUBLAS和CUTLASS库实现非常高的计算资源利用率。然而，整体利用率仍可能受到内核启动开销和GPU核心之间的数据传输的限制，尤其是在除了GeMM之外的不同内核间的数据传输方面。

### Related-works

#### 3D Parallelism

- Data Parallelism：模型副本被复制到多个计算设备（如GPU）上，每个设备处理不同的训练数据子集。每个设备独立计算梯度，然后将梯度进行聚合和平均，以更新模型参数

- Model Parallelism：模型本身被拆分成多个部分，每个部分分配到不同的计算设备上。各个设备负责处理模型的一部分计算

- Pipeline Parallelism：模型的不同层被分配到不同的设备上，训练数据按照一定顺序流过这些设备，就像在流水线上加工产品一样。每个设备只负责计算其负责的那一部分模型

#### ZeRO-Infinity

- 去除冗余Optimizers、gradients、parameters

- 引入CPU和NVMe的分布式内存和存储优化技术，将模型参数、优化器状态和梯度高效地分布在多个设备上，从而减少内存压力

- 采用了一种高效的分片策略，将大规模参数分片存储在多个节点的内存和磁盘上，根据需要动态加载和卸载数据

- 通过Data Parallelism，ZeRO-Infinity 能够高效地利用多设备的计算资源，显著加速训练过程。

- 采用异步通信和重叠计算与通信的方法，最大化计算资源的利用率。

### Strategies on Models

#### Single GPU Transformer Kernels

在小批量情况下优化memory bandwidth，在大批量情况下优化高throughput

#### Many-GPU Dense Transformer Inference System

结合tensor parallelism and pipeline parallelism，在GPU之间高效扩展。

#### Massive-GPU Sparse Model Inference System

结合expert, data,和 tensor parallelism，配合新的通信优化

## Additions

### Memory Occupation:

- Model: Parameters, Gradients, Optimizer

- Training: Activation Memory

kv cache不属于其中任何一种，但类似于Activation Memory


### Activation Memory Optimizing Methods

- Gradient Checkpointing (Activation Checkpointing)：通过在训练过程中只保存部分激活值，减少内存消耗。在需要时重新计算中间激活值，而不是存储所有激活值。这样可以在一定程度上减小内存使用，但会增加额外的计算开销

- Mixed Precision Training：使用半精度浮点数（如FP16）来存储激活值，从而减少内存占用，同时使用全精度浮点数（FP32）进行关键计算，以保持模型的准确性和稳定性

- Memory Compression：通过数据压缩技术对激活值进行压缩存储，减少内存使用。例如，使用稀疏表示或量化技术对激活值进行压缩处理

- 3D Parallelism：Mentioned above
