# LLM Training, Forked by Aarav Gupta
- Applied Scientist @ Microsoft with too much free time

I wanted to dive deeper into different training methodologies, so I forked the original repo from Sasha (original README below), and plan on extending it with additional algorithms to implement in a toy scenario. Then, I want to look at the different metrics (ie peak memory, MFU, completion time) for different model sizes and different world sizes. Pytorch FSDP is very common, for example, but how would it hold up at a larger scale compared to something like ZeRO? Or for something like DDP on a small model, would the additional communication overhead from a more complex technique end up being detrimental to performance at that scale.

Sasha's original puzzle set had:
- Standard Training
- Gradient Accumulation 
    - aggregate loss over minibatches before backprop
- Distributed Data Parallel
    - batches per GPU, stall at backprop for an allreduce
- Weight-Sharded Data Parallel 
    - minibatch forward pass per GPU
    - allgather weights during forward pass, allreduce on backwards
    - weights sharded across GPUs
- Fully Sharded Data Parallel
    - minibatch per GPU
    - allgather weights during forward pass per GPU
    - allgather weights to compute gradients on backward pass
    - scatter reduce gradients on backward pass instead of allreduce 
    - 20% lower peak memory per GPU for same world size, model size compared to WSDP
- Pipeline Parallelism
    - shard layers per GPU so each GPU has an even split of layers
        - 4 GPUs, 8 layers, GPU0=1,2; GPU1=3,4; ...
    - additional communication overhead but has more downstream opportunities for optimization
    - GPU0 processes batch and passes to GPU1, ... to GPU3
    - GPU3 computes loss for layers 7,8, passes to GPU2 for layers 5,6, back to GPU1,0
- GPipe
    - pipeline parallelism but better
    - instead of having bubbles while waiting for other GPUs to send loss back, insert minibatches
    - GPU0 processes minibatch 0, passes to GPU1
    - GPU0 then processes minibatch 1 while GPU1 processes minibatch 0 onwards
    - on and on so that bubbles are minimized and overall throughput is maximized
- Pipeline + FSDP
    - TODO

My Additional Implementations: (todo: better writeups)
- ZeRO
    - earlier paper on scaling model training up to trillion parameter models
- 1F1B
    - interleaving forward-backward passes better more efficiently than GPipe/Pipeline
- Megatron LM
    - linked [here](https://huggingface.co/papers/1909.08053)
- Varuna(?)
    - also interleaving [paper](https://arxiv.org/pdf/2111.04007)
- Will add more too!

# LLM Training Puzzles
- by [Sasha Rush](http://rush-nlp.com) - [srush_nlp](https://twitter.com/srush_nlp) 

![image](https://github.com/srush/LLM-Training-Puzzles/assets/35882/0c46911f-ad1c-4e7a-a42b-2bc2537cccc3)


This is a collection of 8 challenging puzzles about training large language models (or really any NN) on many, many GPUs. 
Very few people actually get a chance to train on thousands of computers, but it is an interesting challenge and one that 
is critically important for modern AI. The goal of these puzzles is to get hands-on experience with the key primitives and to understand 
the goals of memory efficiency and compute pipelining. 


I recommend running in Colab. Click here and copy the notebook to get start.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/srush/LLM-Training-Puzzles/blob/main/puzzles.ipynb)

![image](https://github.com/srush/LLM-Training-Puzzles/assets/35882/6d16fc9e-3d14-4bd0-b7c7-d056e49854ac)



If you are into this kind of thing, this is 6th in a series of these puzzles.

* https://github.com/srush/gpu-puzzles
* https://github.com/srush/tensor-puzzles
* https://github.com/srush/autodiff-puzzles
* https://github.com/srush/transformer-puzzles
* https://github.com/srush/GPTworld
