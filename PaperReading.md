# Paper Reading

## vLLM

### Motivation

现有的llm系统在生成新token时产生的 kv cache 巨大且动态增长和缩减，导致系统

- 产生内部碎片和外部碎片

- 无法进行内存共享

### Paged Attention

将请求的 kv cache 划分为块，每个块可以包含固定数量的令牌的注意力键和值。在 Paged Attention 中，可以像OS的虚拟内存一样更灵活地管理kv cache：可以将block视为page，将token视为byte，将request视为process。进而

- 使用相对较小的块并按需分配来减少内部内存划分

- 由于所有块大小相同，它消除外部内存划分

- 使内存能在block的粒度上，跨不同序列的同一请求甚至跨不同请求共享

### Main Part

- kv cache manager：

    - 在PagedAttention的基础上，将kv cache组织为固定大小的kv block，类似于虚拟内存中的页面。一次请求的kv cache表示为一系列逻辑kv block，随着新token及其kv cache的生成，这些块从左到右逐步填充，最后一个kv block的未填充位置保留用于未来的生成。

    - 在 GPU 工作线程上，块引擎分配一块连续的 GPU DRAM，并将其划分为物理kv block。kv block管理器还维护块表——每个request的逻辑和物理kv block之间的映射。每个块表条目记录逻辑块对应的物理block和已填充位置的数量。将逻辑和物理kv block分开，允许 vLLM 动态增长kv cache内存，而无需提前为所有位置预留，从而消除了现有系统中的大部分内存浪费。

- 调度和抢占：当请求流量超过系统容量时，vLLM必须优先处理一部分请求。在vLLM中，采用先到先服务（FCFS）调度策略处理所有请求，确保公平并防止starvation的产生。当vLLM需要抢占请求时，它确保最早到达的请求优先处理，最新的请求最先被抢占。

- 分布式执行：GPU 工作器不需要在内存管理上进行同步，它们只需要在每个解码迭代的开始时接收所有内存管理信息以及步骤输入。

    - vLLM在处理参数规模过大的情况时使用 Megatron-LM 风格的tensor模型并行策略，这种策略遵循了SPMD的执行模式，其中线性层被分区以进行块状矩阵乘法，并且GPU通过全局归约操作不断同步中间结果。即使使用模型并行执行，每个模型分片仍然处理相同的输入标记集，因此需要相同位置的kv cache。因此，vLLM在集中式调度器内具有单个kv cache管理器。
    
    - 不同的GPU工作器共享该管理器，以及从逻辑块到物理块的映射。这种共享的映射允许GPU工作器使用调度器为每个输入请求提供的物理块执行模型。虽然每个GPU工作器具有相同的物理块ID，但工作器只存储其对应注意力头的kv cache的一部分。
    
    在此后每一步中

    1. 调度器首先为批处理中的每个请求准备包含输入标记 ID 和每个请求的块表的控制消息。
    
    2. 调度器将这个控制消息广播给 GPU 工作器。接着，GPU 工作器开始使用输入标记 ID 执行模型。
    
    3. 在注意力层中，GPU 工作器根据控制消息中的块表读取kv cache。
    
    4. 在执行过程中，GPU 工作器使用全局归约通信原语同步中间结果，无需调度器协调。
    
    5. 最后，GPU 工作器将本次迭代中抽样的标记发送回调度器。

## DeepspeedInference

### Motivation

在生产环境中使用基于Transformer的模型的在线场景需要满足严格的延迟要求，因此使用的批处理大小通常较小。而小批量的性能受限于读取模型权重时的内存带宽利用率，所以需要优化小批量的内存带宽利用率，这面临着三个主要挑战:

- 由于在执行小批量Transformer层操作时，不同内核的工作量有限，推理性能受到内核调用开销的影响。

- 每次内核调用都会将数据写入全局内存，而这些数据在下一次内核调用时被GPU核心读取，这种GPU核心与全局内存之间的数据传输增加了额外的开销。

- cuBLAS和CUTLASS GeMM库对于极小批量的优化不佳，无法实现良好的内存带宽利用率。

大批量推理性能则受到计算资源利用率的限制。在Transformer层内部，像GeMM这样的计算密集型操作可以通过使用CUBLAS和CUTLASS库实现非常高的计算资源利用率。然而，整体利用率仍可能受到内核启动开销和GPU核心之间的数据传输的限制，尤其是在除了GeMM之外的不同内核间的数据传输方面。

### Related Work

#### 3D Parallelism

- Data Parallelism：模型副本被复制到多个计算设备（如GPU）上，每个设备处理不同的训练数据子集。每个设备独立计算梯度，然后将梯度进行聚合和平均，以更新模型参数

- Model Parallelism：模型本身被拆分成多个部分，每个部分分配到不同的计算设备上。各个设备负责处理模型的一部分计算

- Pipeline Parallelism：模型的不同层按层或阶段被划分为多个子模型(stage)，并被分配到不同的设备上，训练数据按照一定顺序流过这些设备，就像在流水线上加工产品一样。每个设备只负责计算其负责的那一部分模型

*Note* 

- Tensor Parallelism: tensor parallelism将模型中的tensor（如权重矩阵）分割成多个部分，并在不同的计算设备上并行计算这些部分，和上述方法有相似但不完全相同

#### ZeRO-Infinity

- 去除冗余Optimizers、gradients、parameters

- 引入CPU和NVMe的分布式内存和存储优化技术，将模型参数、优化器状态和梯度高效地分布在多个设备上，从而减少内存压力

- 采用了一种高效的分片策略，将大规模参数分片存储在多个节点的内存和磁盘上，根据需要动态加载和卸载数据

- 通过Data Parallelism，ZeRO-Infinity 能够高效地利用多设备的计算资源，显著加速训练过程。

- 采用异步通信和重叠计算与通信的方法，最大化计算资源的利用率。

### MoE Computation

- 门控函数：该函数决定每个token分配到的expert。其结果表示为一个稀疏张量，即一个表示序列中每个token被分配到的expert的one-hot向量。

- 稀疏操作序列：包括一个cumsum累加操作符，用于使用前面提到的token 到expert的one-hot 向量计算从expert到 token ID 的逆映射（从expert到 token 的映射）。

- 分散操作符：将 token 分配到相应的expert。该过程实现为一个稀疏的 einsum 操作符，它将前一步计算的expert到 token 的映射与输入 token 进行运算。

- 最终的稀疏 einsum 聚合操作：将每个expert处理过的 token 重新分配回它们的原始顺序。

### Main Part: Strategies on Models

#### Single GPU Transformer Kernels

在小批量情况下优化memory bandwidth，在大批量情况下优化高throughput

- Deep-Fusion：Deep-Fusion通过对计算空间进行切片，避免全局同步，实现了不仅元素级操作，还包括归约、数据转置和GeMM的融合。这样可以减少内核启动和数据传输开销，并在寄存器或共享内存中重用数据

- SBI-GeMM：自定义GeMM内核通过切片策略、协作组归约和数据布局转换提高内存带宽利用率。通过在共享内存中进行数据转置，使部分结果在内存中连续，可以减少warp级同步带来的性能瓶颈。此外，初始化时转置权重矩阵，使读取时能够充分利用缓存行，提高了内存带宽利用率

#### Many-GPU Dense Transformer Inference System

结合tensor parallelism (Memory Bandwidth) and pipeline parallelism (Memory Volume)，在GPU之间高效扩展。三个优化措施：

- Hiding data dependencies and hybrid scheduling：

    1. 将要推理的一批batchsize为B的序列{$ s_1$, $ s_2$, ..., $ s_B$}分成微批次组，每个微批次是提供给每个内核调用的计算单元

    2. 微批次在模型流水线的各个阶段之间推进，直到最后一个阶段产生下一个token,如果序列$s_i$未终止，生成的token将用作生成下一个token的输入

    3. 为了充分利用流水线的所有阶段，将微批次数量设置为流水线深度P避免由于较大批次带来的延迟和内存开销

    4. 动态排队生成的token微批次直到序列终止，避免了中间的流水线气泡

    5. 在提示处理阶段和token生成阶段采用不同数量的微批次，以实现最佳性能。提示处理阶段使用较多的微批次以最小化流水线气泡，而在token生成阶段减少微批次数量以降低总体执行时间

- Offloading Activations to CPU Memory：当分配的激活内存超过阈值时，将部分激活从GPU转移到CPU内存，从而释放GPU内存以容纳更大的批次并提高系统利用率

- Communication Optimization： 为了避免因CPU内存转移带来的通信瓶颈，重叠通信与计算，并采用架构感知的通信优化策略，奇数编号的GPU卸载奇数层的激活，偶数编号的GPU卸载偶数层的激活，防止在PCIe链路上的争用，充分利用PCIe带宽

#### Massive-GPU Sparse Model Inference System

针对MoE结合expert, data,和 tensor parallelism，配合新的通信优化：

1. 使用table-data structure 替换 one-hot表示token to expert mapping，避免one-hot的0值，降低内存开销

2. 基于token-to-expert mapping table创建逆映射

3. 使用data-layout transformation替换sparse einsum based scatter operation，其过程为先使用expert-to-token mapping table确定分配给expert的token ID，然后将这些token复制到相应的expert位置

4. 使用data-layout transformation替换sparse einsum based gather operation，过程类似step 3

## Nuggets

### Motivation

过往用于模型fine-tuning的训练数据主要依赖经验方法，包括应用启发式规则、人类审查以及基于模型性能反馈的迭代数据调整。这种trial-and-error method的训练消耗了大量的人力和计算资源。

### Related Work

#### One-shot Learning

单样本学习是一种机器学习方法，旨在通过仅用一个或极少的样本来进行有效的分类或识别任务。与传统的机器学习方法需要大量数据进行训练不同，单样本学习能够在数据极其稀少的情况下进行有效的学习和预测

- 少样本学习（Few-Shot Learning）：单样本学习是少样本学习的一种特殊情况，少样本学习包括更多的情况，例如 5-shot 学习（即每个类别有5个样本）

- 泛化能力：模型不仅要能够记住已有的少量样本，还要能够推广到新的、未见过的样本上

#### Instruction Tuning

将NLP任务转化为自然语言instruction用于训练，以提高LLM在zero shot任务上的表现

### Main Part

提出：

- Zero Shot Score：评估LLMs在没有先前示例的情况下执行各种任务的能力，生成基准性能分数

- One Shot Score：使用单个示例作为instruction，重新评估模型在相同任务上的表现，提供新的分数

- Golden Score(GS)：Definited as $GS(z) = \frac{1}{m} \sum_{i = 1}^{m} \mathbb{I}[s_{iit}^i(z)>s_{zsl}^i(z)] \in [0, 1], where z as a example of instruction, m as the number of tasks, s_{iit} as One Shot Score, s_{zsl} as Zero Shot Score$，认为GS越大，该instruction越有价值

对instruction数据集设定GS阈值进行筛选得到优质训练数据即为Nuggets

实验发现只使用好的instruction训练效果可能比使用完整instruction集更好，但GS过高筛选剩余的instruction数量过少又会使训练效果下降，因此需要考虑optimal GS阈值。

可从预定义任务集选择一个子集来得到instruction集的GS值。实验发现使用KMeans对预定义任务聚类后，选择聚类质心作为该子集得到的GS值应用效果更佳。

## Voyager

### Related Work

#### Chain of Thoughts

通过生成一系列中间推理步骤（链式推理），并在提示中提供一些示例，这种推理能力可以自然地在足够大的语言模型中出现。

- 链式推理提示在一系列算术、常识和符号推理任务中显著提高了模型性能

- 链式推理提示对较小模型无显著影响，但在模型参数达到约100B时，才会显著提升模型性能。较小规模的模型尽管生成了流畅的链式推理，但逻辑性较差，导致表现不佳​

### Main Part

Voyager为一种在开放世界环境中自动进行探索和任务执行的智能体系统。该系统基于大型语言模型（如GPT-4），通过自动化任务分解和自我验证模块，提升了探索和任务完成能力。主要包括三个组成部分：

- Automatic Curriculum：系统自动生成一系列逐步增加难度的任务，形成一个课程体系，推动智能体持续学习和进步，使Voyager能够在无需人为干预的情况下进行任务学习和技能提升

- Skill Library：Voyager在探索过程中积累技能，保存在技能库中。在需要时，系统从技能库中检索并应用已学得的技能，提高任务执行的效率和准确性

- Iterative Prompting Mechanism：为行为生成可执行代码，并通过自我检测和环境或编译器反馈来提高代码生成的效率和准确性

## Additions

### Memory Architecture of CPU and GPU

In the order of architechture hierarchy or access speed

#### CPU

- Register：最靠近处理核心，速度最快，用于存储正在处理的数据和指令

- L1 Cache：紧邻CPU核心，容量较小但速度极快，通常分为指令缓存和数据缓存

- L2 Cache：容量更大，但速度稍慢，常常为每个核心独立配置

- L3 Cache：多个核心共享，容量更大但速度较慢，减少不同核心间的数据传输延迟

- RAM：容量较大，存取速度较慢，用于存储操作系统、应用程序和当前处理的数据

- Storage：硬盘（HDD）或固态硬盘（SSD），存储大量的持久化数据，速度最慢

#### GPU

- Register：每个计算单元（流处理器）拥有大量寄存器，用于存储当前正在处理的数据

- Shared Memory：一个线程块内的所有线程共享，速度较快，适合需要高频访问的数据

- L1 and L2 Cache：类似于CPU，但结构和配置更适应并行计算，L2缓存通常在多个计算单元间共享

- Global Memory：GPU的主要存储器，容量大但速度较慢，所有线程都可以访问

- Constant and Texture Memory：用于存储只读数据，优化特定类型的数据访问模式

### Memory Occupation:

- Model: Parameters, Gradients, Optimizer

- Training: Activation Memory

kv cache不属于其中任何一种，但类似于Activation Memory


### Activation Memory Optimizing Methods

- Gradient Checkpointing (Activation Checkpointing)：通过在训练过程中只保存部分激活值，减少内存消耗。在需要时重新计算中间激活值，而不是存储所有激活值。这样可以在一定程度上减小内存使用，但会增加额外的计算开销

- Mixed Precision Training：使用半精度浮点数（如FP16）来存储激活值，从而减少内存占用，同时使用全精度浮点数（FP32）进行关键计算，以保持模型的准确性和稳定性

- Memory Compression：通过数据压缩技术对激活值进行压缩存储，减少内存使用。例如，使用稀疏表示或量化技术对激活值进行压缩处理

- 3D Parallelism：Mentioned above

### Prompt Components

- Instruction：Instructions are likely the most commonly used prompt component

- Primary Content：Primary content refers to some sort of text that is being processed or transformed by the model

- Examples：Giving Examples to the model

- Cue：Cues act as the "jumpstart" for the output of the model, helping to direct the model to the desired output

- Supporting Content：Supporting content is information that the model can utilize to influence the output in some way
