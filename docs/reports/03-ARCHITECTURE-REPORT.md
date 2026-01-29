# RuVector Architecture Report

## Executive Summary

RuVector implements a four-layer architecture designed for high-performance vector operations with self-learning capabilities. This document provides a comprehensive technical overview of the system architecture, component interactions, and design decisions.

---

## System Architecture Overview

```mermaid
flowchart TB
    subgraph "Client Layer"
        REST[REST API]
        NAPI[Node.js NAPI]
        WASM[WASM Module]
        CLI[CLI Interface]
        MCP[MCP Server]
        PG[PostgreSQL Extension]
    end

    subgraph "Application Layer"
        VDB[VectorDB API]
        ADB[AgenticDB API]
        SONA[SONA Router]
    end

    subgraph "Query Engine"
        QP[Query Processor]
        ATT[Attention Mechanisms]
        HYB[Hybrid Search]
        FLT[Filter Engine]
    end

    subgraph "Index Layer"
        HNSW[HNSW Index]
        GNN[GNN Learning Layer]
        QUANT[Quantization]
        FLAT[Flat Index]
    end

    subgraph "Storage Layer"
        REDB[(redb Store)]
        MMAP[Memory-Mapped Files]
        RKYV[rkyv Serialization]
    end

    subgraph "Distributed Layer"
        RAFT[Raft Consensus]
        REP[Replication]
        SHARD[Sharding]
    end

    REST --> VDB
    NAPI --> VDB
    WASM --> VDB
    CLI --> VDB
    MCP --> VDB
    PG --> VDB

    VDB --> QP
    ADB --> QP
    SONA --> QP

    QP --> HNSW
    ATT --> HNSW
    HYB --> HNSW
    FLT --> HNSW

    HNSW <--> GNN
    HNSW --> QUANT
    QUANT --> MMAP
    HNSW --> FLAT

    HNSW --> REDB
    MMAP --> RKYV

    HNSW --> RAFT
    RAFT --> REP
    REP --> SHARD
```

---

## Layer 1: Storage Layer

### Purpose
Persistent, crash-safe storage with zero-copy access patterns.

### Components

#### 1.1 redb (ACID Metadata Store)

```mermaid
flowchart LR
    subgraph "redb Structure"
        T1[vectors_meta table]
        T2[config table]
        T3[agenticdb tables]
        T4[index_meta table]
    end

    subgraph "Guarantees"
        G1[ACID transactions]
        G2[Crash recovery]
        G3[Concurrent reads]
    end

    T1 --> G1
    T2 --> G1
    T3 --> G2
    T4 --> G3
```

**Data Stored**:
- Vector IDs and metadata
- HNSW configuration
- AgenticDB 5-table schema
- Index build parameters

#### 1.2 Memory-Mapped Vectors (memmap2)

```mermaid
flowchart TD
    subgraph "File Structure"
        F1[vectors.bin file]
    end

    subgraph "Memory Layout"
        M1[Header: dimensions, count]
        M2[Vector 0: f32 × dim]
        M3[Vector 1: f32 × dim]
        M4[...]
        M5[Vector N: f32 × dim]
    end

    subgraph "Access Pattern"
        A1[OS page cache]
        A2[Zero-copy read]
        A3[Lazy loading]
    end

    F1 --> M1
    M1 --> M2
    M2 --> M3
    M3 --> M4
    M4 --> M5

    M2 --> A1
    A1 --> A2
    A2 --> A3
```

**Benefits**:
- Supports datasets larger than RAM
- No deserialization overhead
- OS manages caching

#### 1.3 rkyv (Zero-Copy Serialization)

```mermaid
flowchart LR
    subgraph "Traditional"
        T1[Serialize] --> T2[Disk]
        T2 --> T3[Deserialize]
        T3 --> T4[Use]
    end

    subgraph "rkyv"
        R1[Archive] --> R2[Disk]
        R2 --> R3[Validate]
        R3 --> R4[Direct access]
    end

    style T3 fill:#f99
    style R3 fill:#9f9
```

**Performance**:
- Sub-second index loading for billions of vectors
- In-place access without copying
- Validation-only overhead

---

## Layer 2: Index Layer

### Purpose
Fast approximate nearest neighbor (ANN) search with learning capability.

### Components

#### 2.1 HNSW (Hierarchical Navigable Small World)

```mermaid
flowchart TD
    subgraph "HNSW Layers"
        L3[Layer 3: Sparse entry points]
        L2[Layer 2: Medium density]
        L1[Layer 1: Higher density]
        L0[Layer 0: All vectors]
    end

    subgraph "Search Process"
        S1[Enter at top layer]
        S2[Greedy descent]
        S3[Refine at layer 0]
        S4[Return k-nearest]
    end

    L3 --> L2
    L2 --> L1
    L1 --> L0

    S1 --> S2
    S2 --> S3
    S3 --> S4
```

**Parameters**:

| Parameter | Description | Default | Impact |
|-----------|-------------|---------|--------|
| M | Connections per node | 32 | Memory vs recall |
| ef_construction | Build-time candidates | 200 | Build speed vs quality |
| ef_search | Search-time candidates | 100 | Latency vs recall |

**Complexity**:
- Build: O(n × log(n))
- Search: O(log(n))
- Memory: ~640 bytes per vector (M=32, 128D)

#### 2.2 GNN Learning Layer

```mermaid
flowchart TD
    subgraph "Learning Pipeline"
        Q[User Query] --> R[Search Results]
        R --> F[User Feedback]
        F --> E[Experience Buffer]
        E --> G[GNN Processing]
        G --> W[Weight Updates]
        W --> H[HNSW Graph Updated]
    end

    subgraph "GNN Architecture"
        I[Input: Node embeddings]
        A[Graph Attention]
        M[Message Passing]
        O[Output: Edge weights]
    end

    G --> I
    I --> A
    A --> M
    M --> O
    O --> W
```

**Learning Mechanism**:
1. Observe which results users click/rate
2. Store as (query, positive, negative) tuples
3. GNN learns to predict edge importance
4. HNSW graph edges reweighted
5. Future searches improved

#### 2.3 Quantization

```mermaid
flowchart LR
    subgraph "Original"
        O1[32-bit float × 128]
        O2[512 bytes/vector]
    end

    subgraph "Scalar Quantization"
        S1[8-bit int × 128]
        S2[128 bytes/vector]
        S3[4× compression]
    end

    subgraph "Product Quantization"
        P1[8 subspaces × 8-bit]
        P2[8 bytes/vector]
        P3[64× compression]
    end

    subgraph "Binary"
        B1[1-bit × 128]
        B2[16 bytes/vector]
        B3[32× compression]
    end

    O1 --> S1
    O1 --> P1
    O1 --> B1
```

---

## Layer 3: Query Engine

### Purpose
Advanced query processing with multiple search strategies.

### Components

#### 3.1 Distance Metrics (SIMD-Optimized)

```mermaid
flowchart TD
    subgraph "Available Metrics"
        E[Euclidean L2]
        C[Cosine Similarity]
        D[Dot Product]
        M[Manhattan L1]
    end

    subgraph "SIMD Acceleration"
        S1[Detect CPU features]
        S2{AVX-512?}
        S3{AVX2?}
        S4{SSE?}
        S5[Scalar fallback]

        S1 --> S2
        S2 -->|Yes| A1[16× speedup]
        S2 -->|No| S3
        S3 -->|Yes| A2[8× speedup]
        S3 -->|No| S4
        S4 -->|Yes| A3[4× speedup]
        S4 -->|No| S5
    end

    E --> S1
    C --> S1
    D --> S1
    M --> S1
```

#### 3.2 Attention Mechanisms (39 Types)

```mermaid
mindmap
    root((Attention))
        Standard
            Multi-Head
            Flash Attention
            Grouped Query
        Graph-Based
            GATv2
            SoftPool
            EdgeConv
        Geometric
            Hyperbolic
            Spherical
            Mixed-Curvature
        Sparse
            Mincut-Gated
            Top-K
            Random
        Neuromorphic
            Spiking
            BTSP
            Homeostatic
        Quantum
            Coherence-Gated
            Superposition
```

#### 3.3 Hybrid Search

```mermaid
flowchart TD
    subgraph "Query Processing"
        Q[User Query] --> KW[Keyword Extraction]
        Q --> EM[Embedding Generation]
    end

    subgraph "Parallel Search"
        KW --> BM25[BM25 Keyword Search]
        EM --> VEC[Vector Semantic Search]
    end

    subgraph "Fusion"
        BM25 --> RRF[Reciprocal Rank Fusion]
        VEC --> RRF
        RRF --> MMR[Maximal Marginal Relevance]
        MMR --> OUT[Diverse Results]
    end
```

---

## Layer 4: Application Layer

### Purpose
User-facing APIs and AI integration.

### Components

#### 4.1 VectorDB API

```mermaid
classDiagram
    class VectorDB {
        +new(options: DbOptions) VectorDB
        +insert(entry: VectorEntry) VectorId
        +insert_batch(entries: Vec~VectorEntry~) Vec~VectorId~
        +search(query: SearchQuery) Vec~SearchResult~
        +delete(id: VectorId) ()
        +count() usize
    }

    class DbOptions {
        +dimensions: usize
        +distance_metric: DistanceMetric
        +storage_path: String
        +hnsw_config: Option~HnswConfig~
        +quantization: Option~QuantizationConfig~
    }

    class VectorEntry {
        +id: Option~VectorId~
        +vector: Vec~f32~
        +metadata: Option~Metadata~
    }

    class SearchQuery {
        +vector: Vec~f32~
        +k: usize
        +filter: Option~Filter~
        +ef_search: Option~usize~
    }

    VectorDB --> DbOptions
    VectorDB --> VectorEntry
    VectorDB --> SearchQuery
```

#### 4.2 AgenticDB Schema

```mermaid
erDiagram
    VECTORS {
        string id PK
        float[] embedding
        json metadata
        timestamp created
    }

    REFLEXION_EPISODES {
        string id PK
        string thought
        string action
        string observation
        string critique
        float[] embedding
        timestamp created
    }

    SKILLS_LIBRARY {
        string id PK
        string name
        string description
        float[] embedding
        json parameters
        float success_rate
        int usage_count
    }

    CAUSAL_EDGES {
        string id PK
        string from_id FK
        string to_id FK
        string relationship
        float weight
        timestamp observed
    }

    LEARNING_SESSIONS {
        string session_id PK
        string agent_id
        json experiences
        float total_reward
        timestamp started
        timestamp ended
    }

    VECTORS ||--o{ CAUSAL_EDGES : connects
    REFLEXION_EPISODES ||--o{ SKILLS_LIBRARY : consolidates
    SKILLS_LIBRARY ||--o{ LEARNING_SESSIONS : used_in
```

#### 4.3 SONA Self-Optimizing Router

```mermaid
flowchart TD
    subgraph "Two-Tier Architecture"
        L1[Layer 1 LoRA: Fast patterns]
        L2[Layer 2 LoRA: Deep analysis]
    end

    subgraph "Routing Decision"
        Q[Query] --> F[Feature extraction]
        F --> C{Cached pattern?}
        C -->|Yes| L1
        C -->|No| L2
        L1 --> R1[Fast route <1ms]
        L2 --> R2[Analyzed route <50ms]
    end

    subgraph "Learning"
        R1 --> O[Observe outcome]
        R2 --> O
        O --> EWC[EWC++ update]
        EWC --> L1
    end
```

---

## Distributed Architecture

### Raft Consensus

```mermaid
stateDiagram-v2
    [*] --> Follower
    Follower --> Candidate: Election timeout
    Candidate --> Leader: Wins election
    Candidate --> Follower: Higher term seen
    Leader --> Follower: Higher term seen

    state Leader {
        [*] --> AcceptWrites
        AcceptWrites --> Replicate
        Replicate --> WaitQuorum
        WaitQuorum --> Commit
        Commit --> AcceptWrites
    }
```

### Sharding Strategy

```mermaid
flowchart TD
    subgraph "Hash-Based Sharding"
        V[Vector ID] --> H[Hash Function]
        H --> M[Modulo N shards]
        M --> S[Shard assignment]
    end

    subgraph "Shard Distribution"
        S --> S1[Shard 0: Node A]
        S --> S2[Shard 1: Node B]
        S --> S3[Shard 2: Node C]
    end

    subgraph "Replication"
        S1 --> R1[Replica on Node B]
        S2 --> R2[Replica on Node C]
        S3 --> R3[Replica on Node A]
    end
```

---

## Crate Dependency Graph

```mermaid
flowchart TD
    subgraph "Core"
        C[ruvector-core]
    end

    subgraph "Bindings"
        N[ruvector-node]
        W[ruvector-wasm]
        CLI[ruvector-cli]
        PG[ruvector-postgres]
    end

    subgraph "Advanced Features"
        G[ruvector-graph]
        GNN[ruvector-gnn]
        ATT[ruvector-attention]
        MC[ruvector-mincut]
    end

    subgraph "Distributed"
        RAFT[ruvector-raft]
        REP[ruvector-replication]
        CLUS[ruvector-cluster]
    end

    subgraph "Learning"
        SONA[sona]
        ROUTE[ruvector-router-core]
        LEARN[ruvector-learning-wasm]
    end

    subgraph "LLM"
        LLM[ruvllm]
        LLMC[ruvllm-cli]
        LLMW[ruvllm-wasm]
    end

    N --> C
    W --> C
    CLI --> C
    PG --> C

    G --> C
    GNN --> G
    ATT --> C
    MC --> C

    RAFT --> C
    REP --> RAFT
    CLUS --> REP

    SONA --> C
    ROUTE --> SONA
    LEARN --> ROUTE

    LLM --> C
    LLMC --> LLM
    LLMW --> LLM
```

---

## Performance Architecture

### Memory Layout

```mermaid
flowchart TD
    subgraph "Per-Vector Memory"
        V[Vector Data]
        M[Metadata Pointer]
        E[HNSW Edges]
        Q[Quantized Copy]
    end

    subgraph "Index Memory"
        L0[Layer 0 Graph]
        L1[Layer 1 Graph]
        LN[Layer N Graph]
        EP[Entry Points]
    end

    subgraph "Cache"
        VC[Vector Cache]
        MC[Metadata Cache]
        QC[Query Cache]
    end
```

**Memory Formula**:
```
Total Memory =
    vectors × dimensions × sizeof(f32) +          # Raw vectors
    vectors × 640 bytes × M/32 +                  # HNSW graph
    vectors × metadata_avg_size +                 # Metadata
    cache_size                                    # Caches
```

### Parallel Processing

```mermaid
flowchart LR
    subgraph "Batch Insert"
        B1[Split into chunks]
        B2[Parallel embedding validation]
        B3[Concurrent HNSW insertion]
        B4[Batch commit]
    end

    subgraph "Parallel Search"
        S1[Query arrives]
        S2[HNSW traversal single-thread]
        S3[Parallel candidate scoring]
        S4[Merge results]
    end

    B1 --> B2 --> B3 --> B4
    S1 --> S2 --> S3 --> S4
```

---

## Security Architecture

```mermaid
flowchart TD
    subgraph "Input Validation"
        I1[Dimension check]
        I2[Metadata sanitization]
        I3[Path traversal prevention]
    end

    subgraph "Execution Safety"
        E1[Constant-time operations]
        E2[Bounds checking]
        E3[Memory safety Rust]
    end

    subgraph "Data Protection"
        D1[ACID transactions]
        D2[Encryption at rest optional]
        D3[Access control optional]
    end

    Input --> I1
    I1 --> I2
    I2 --> I3
    I3 --> E1
    E1 --> E2
    E2 --> E3
    E3 --> D1
    D1 --> D2
    D2 --> D3
```

---

## Design Decisions

### Why Rust?

| Decision | Rationale |
|----------|-----------|
| Memory safety | No buffer overflows, use-after-free |
| Zero-cost abstractions | High-level APIs without overhead |
| Fearless concurrency | Data race prevention at compile time |
| FFI support | Easy Node.js/WASM/Python bindings |
| Performance | C/C++ level speed |

### Why HNSW over alternatives?

| Index Type | Search | Build | Memory | Chosen? |
|------------|--------|-------|--------|---------|
| HNSW | O(log n) | O(n log n) | Medium | **Yes** |
| IVF | O(√n) | O(n) | Low | No |
| LSH | O(1) | O(n) | High | No |
| KD-Tree | O(log n) | O(n log n) | Low | No |

HNSW provides the best balance of search speed, recall, and dynamic updates.

### Why rkyv over serde?

| Feature | rkyv | serde |
|---------|------|-------|
| Deserialization | Zero-copy | Full copy |
| Access speed | Immediate | After parse |
| Memory overhead | None | Full object |
| Validation | Optional | Required |

---

## Summary

RuVector's architecture is designed around three key principles:

1. **Performance**: SIMD, zero-copy, parallel processing
2. **Learning**: GNN layer on HNSW enables continuous improvement
3. **Flexibility**: Multiple deployment targets from single codebase

The four-layer design (Storage → Index → Query → Application) provides clean separation of concerns while enabling tight integration for maximum performance.
