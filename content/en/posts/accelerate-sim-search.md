+++
date = '2026-01-15'
draft = false
title = 'Accelerating Similarity Search with Work Distribution, Memory Coalescing, and Shared Memory'
+++

## Introduction

In the age of large language models and retrieval-augmented generation (RAG), similarity search has become a fundamental operation. Whether you're building a semantic search engine, a recommendation system, or a RAG pipeline, you'll often need to find the most similar vectors from a large database given a query vector. But what happens when your database contains millions of vectors? Performance becomes critical in building these systems.

In this article, i will demonsrate how GPU optimization techniques such as work distribution, memory coalescing, and shared memory accelerate similarity search operations. We'll walk through CUDA implementations, demonstrate the performance gains, and explain exactly why these optimizations work.

## Similarity Search at Scale

Similarity search is the task of finding the most similar items in a database to a given query. In the context of vector embeddings, this typically means computing a similarity metric between a query vector and millions of database vectors, then finding the highest-scoring matches.

Consider a typical RAG application: when a user asks a question, the system needs to:
1. Convert the question to a vector embedding
2. Compare it against millions of document embeddings
3. Retrieve the top-k most relevant documents
4. Feed them to your language model

Computing similarity against millions of vectors can easily become a bottleneck. On a CPU, processing 10,000 queries against 1 million database vectors (each with 128 dimensions) involves roughly 1.28 billion dot products. GPUs in other hand, has parallel cores that can be used to compute many dot products simultaneously. However, we still need to implement the memory access patterns optimization in order to utilize said capability. 

## Experimental Setup

**Hardware:**
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU
- CPU: 13th Gen Intel(R) Core(TM) i9-13900HX
- RAM: 32 GB

**Software:**
- Python 3.11.13
- Nvidia Nsight Systems
- Numba 0.63.1
- NumPy 2.3.5

**Dataset:**
We'll use the SIFT1M dataset, a standard benchmark in the similarity search task:
- **Database vectors**: 1,000,000 vectors × 128 dimensions
- **Query vectors**: 10,000 vectors × 128 dimensions
- **Ground truth**: Pre-computed top-100 nearest neighbors
- **Total size**: 576.8 MB
## Naive GPU Implementation

Each thread processes one query, computing its similarity against all database vectors:

```python
import numpy as np
import numba
from numba import cuda

@cuda.jit
def similarity_search_non_coalesced(queries, database, results):
    query_idx = cuda.grid(1)
    if query_idx < queries.shape[0]:
        n_db = database.shape[0]
        dim = database.shape[1]
        best_sim = -1e9
        best_idx = -1

        for db_idx in range(n_db):
            sim = 0.0
            for feat_idx in range(dim):
                sim += queries[query_idx, feat_idx] * database[db_idx, feat_idx]

            if sim > best_sim:
                best_sim = sim
                best_idx = db_idx

        results[query_idx, 0] = best_idx
        results[query_idx, 1] = best_sim
```

1. Each thread is assigned one query via `cuda.grid(1)`
2. For each query, iterate through all database vectors
3. For each database vector, compute the dot product
4. Track the best similarity score and its index
5. Store the results

Before doing the benchmark and optimizing the access patterns, we will need to understand Memory Coalescing and Shared Memory.
## Memory Coalescing

### The GPU Memory Hierarchy

GPUs have multiple levels of memory:
- **Global memory**: Largest and most abundant memory available (GBs), but slowest to access 
- **Shared memory**: On-chip memory with much faster speed than global memory 
- **Registers**: Fastest memory type located on the Streaming Multiprocessor

Accessing global memory can be an expensive operation, but when threads in a warp (a group of 32 threads that execute in lockstep) access consecutive memory addresses, the GPU can coalesce multiple memory requests into a single transaction.

### What is Memory Coalescing?

Memory coalescing occurs when adjacent threads access adjacent memory locations. Instead of issuing 32 separate memory requests, the GPU can issue a single request that fetches all required data in one transaction.

Let's take a look at this line in the naive implementation:
```python
sim += queries[query_idx, feat_idx] * database[db_idx, feat_idx]
```

While threads access the same `db_idx` at the same time, the memory layout of a row-major array means that `database[db_idx, feat_idx]` and `database[db_idx, feat_idx+1]` are consecutive in memory, but `database[db_idx, feat_idx]` and `database[db_idx+1, feat_idx]` are far apart due to the separation by an entire row with 128 dimension.
## Understanding Shared Memory

Shared memory is a memory space accessible by all threads within a single thread block stored on L1 Data Cache of GPU's Streaming Multiprocessor (SM). Through the usage of *shared memory*, the need for all threads to write to global  memory repeatedly can be avoided.
## The Optimized Implementation

Now let's see how we can combine memory coalescing and shared memory to improve performance:

```python
import numpy as np
import numba
from numba import cuda

@cuda.jit
def similarity_search_true_coalesced(queries, database_T, results):
    query_idx = cuda.blockIdx.x
    thread_idx = cuda.threadIdx.x
    block_size = cuda.blockDim.x

    if query_idx >= queries.shape[0]:
        return

    dim = database_T.shape[0]
    n_db = database_T.shape[1]

    shared_best_sim = cuda.shared.array(256, dtype=numba.float32)
    shared_best_idx = cuda.shared.array(256, dtype=numba.int32)

    local_best_sim = -1e9
    local_best_idx = -1

    for db_idx in range(thread_idx, n_db, block_size):
        sim = 0.0
        for feat_idx in range(dim):
            # COALESCED: database_T[feat_idx, db_idx]
            # Adjacent threads access consecutive memory locations
            # thread 0: db_idx=0, thread 1: db_idx=1, etc.
            sim += queries[query_idx, feat_idx] * database_T[feat_idx, db_idx]

        if sim > local_best_sim:
            local_best_sim = sim
            local_best_idx = db_idx

    shared_best_sim[thread_idx] = local_best_sim
    shared_best_idx[thread_idx] = local_best_idx
    cuda.syncthreads()

    # Reduction in shared memory
    stride = block_size // 2
    while stride > 0:
        if thread_idx < stride:
            if shared_best_sim[thread_idx + stride] > shared_best_sim[thread_idx]:
                shared_best_sim[thread_idx] = shared_best_sim[thread_idx + stride]
                shared_best_idx[thread_idx] = shared_best_idx[thread_idx + stride]
        cuda.syncthreads()
        stride //= 2

    if thread_idx == 0:
        results[query_idx, 0] = shared_best_idx[0]
        results[query_idx, 1] = shared_best_sim[0]
```

## Differences Between the Two Implementations

### 1. Thread Organization
**First Implementation:**
```python
query_idx = cuda.grid(1)  # Each thread = one query
```

**Optimized Implementation:**
```python
query_idx = cuda.blockIdx.x   # Each block = one query
thread_idx = cuda.threadIdx.x  # Threads cooperate on one query
```

In the optimized version, we use an entire block of threads to process a single query.
### 2. Work Distribution
**First Implementation:**
```python
for db_idx in range(n_db):  # Each thread processes all vectors
```

**Optimized Implementation:**
```python
for db_idx in range(thread_idx, n_db, block_size):  # Threads split the work
```

Instead of each thread processing all database vectors, threads in a block divide them up. Thread 0 processes vectors 0, 256, 512, ..., while thread 1 processes 1, 257, 513, ..., and so on.
### 3. Memory Coalescing Implementation
**First Implementation:**
```python
database[db_idx, feat_idx]  # Shape: (n_vectors, dim)
```

**Optimized Implementation:**
```python
database_T[feat_idx, db_idx]  # Shape: (dim, n_vectors)
```

By transposing the database from `(n_vectors, dim)` to `(dim, n_vectors)`, we ensure that when adjacent threads access `database_T[feat_idx, 0]`, `database_T[feat_idx, 1]`, `database_T[feat_idx, 2]`, etc., they're accessing consecutive memory locations.
### 4. Shared Memory Implementation Details

The shared memory optimization is implemented in line 36-48:
#### Step 1: Declare Shared Memory (lines 18-19)
```python
shared_best_sim = cuda.shared.array(256, dtype=numba.float32)
shared_best_idx = cuda.shared.array(256, dtype=numba.int32)
```

First, we allocate two arrays in shared memory, one for similarity scores, one for indices. These arrays are visible to all 256 threads in the block.
#### Step 2: Store Local Results (lines 36-37)
```python
shared_best_sim[thread_idx] = local_best_sim
shared_best_idx[thread_idx] = local_best_idx
```

After each thread finds its best match among its assigned database vectors, it stores the result in shared memory. Thread 0 writes to position 0, thread 1 to position 1, etc.
#### Step 3: Synchronize Threads (line 38)
```python
cuda.syncthreads()
```

This ensures all threads have finished writing before we proceed
#### Step 4: Parallel Reduction (lines 41-48)
```python
stride = block_size // 2
while stride > 0:
    if thread_idx < stride:
        if shared_best_sim[thread_idx + stride] > shared_best_sim[thread_idx]:
            shared_best_sim[thread_idx] = shared_best_sim[thread_idx + stride]
            shared_best_idx[thread_idx] = shared_best_idx[thread_idx + stride]
    cuda.syncthreads()
    stride //= 2
```

To understand parallel reduction algorithm, Imagine 256 values that we want to reduce to a single maximum:

**Iteration 1 (stride=128):**
- Thread 0 compares positions 0 and 128, keeps the max
- Thread 1 compares positions 1 and 129, keeps the max
- ...
- Thread 127 compares positions 127 and 255, keeps the max
- Now we have 128 candidates

**Iteration 2 (stride=64):**
- Thread 0 compares positions 0 and 64, keeps the max
- Thread 1 compares positions 1 and 65, keeps the max
- ...
- Now we have 64 candidates

This continues (32, 16, 8, 4, 2, 1) until thread 0 holds the global maximum. Instead of having one thread sequentially check 256 values (256 steps), the reduction reduced the step to log₂(256) = 8 parallel steps.
#### Step 5: Write Final Result (lines 50-52)
```python
if thread_idx == 0:
    results[query_idx, 0] = shared_best_idx[0]
    results[query_idx, 1] = shared_best_sim[0]
```

Only thread 0 writes the final result back to global memory

## Benchmark Results

| Implementation            | Time (ms) | Queries/sec | Speedup |
| ------------------------- | --------- | ----------- | ------- |
| Non-coalesced             | 33500.43  | 299         | 1.0x    |
| Coalesced + Shared Memory | 20475.88  | 488         | 1.64x   |

We can see improvement in the Coalesced + Shared Memory based on the time it took to complete the operation. Comparing both operations using NVIDIA Nsight Systems gives more details:

![coalesced](/images/coalesced.jpg)

![non-coalesced](/images/non-coalesced.jpg)

Unallocated warps percentage can be seen to reach 80% in the first implementation. After distributing the workload and implementing said optimization techniques in the coalesced version, the warp slots available has been maximized in the SM up to >99%.
## Conclusion

In this article, we demonstrated two GPU optimization techniques:

1. **Thread Organizing and Work Distribution**: Workload shift from one thread towards block of threads, which eliminates the serial nature of one thread-one query that becomes the bottleneck in demonstrated similarity search operation

2. **Memory Coalescing**: Organizing data and thread access patterns done by the *thread* so that separated transactions of writing memory can be coalesced to one.

3. **Shared Memory**: Using a memory space that can be accessed by all threads in a block in order to avoid the need for threads to write to global memory repeatedly

These optimizations yielded an 1.64x speedup on the SIFT1M benchmark which can be beneficial for a production system that also uses similarity search operations.