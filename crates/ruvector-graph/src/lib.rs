//! # RuVector Graph Database
//!
//! A high-performance graph database layer built on RuVector with Neo4j compatibility.
//! Supports property graphs, hypergraphs, Cypher queries, ACID transactions, and distributed queries.

pub mod error;
pub mod types;
pub mod property;
pub mod node;
pub mod edge;
pub mod hyperedge;
pub mod index;
pub mod storage;
pub mod graph;
pub mod cypher;
pub mod transaction;
pub mod executor;

// Performance optimization modules
pub mod optimization;

// Vector-graph hybrid query capabilities
pub mod hybrid;

// Distributed graph capabilities
#[cfg(feature = "distributed")]
pub mod distributed;

// Core type re-exports
pub use error::{GraphError, Result};
pub use types::{NodeId, EdgeId, PropertyValue, Properties, Label, RelationType};
pub use node::{Node, NodeBuilder};
pub use edge::{Edge, EdgeBuilder};
pub use hyperedge::{Hyperedge, HyperedgeBuilder, HyperedgeId};
pub use graph::GraphDB;
pub use storage::GraphStorage;
pub use transaction::{Transaction, TransactionManager, IsolationLevel};

// Re-export hybrid query types when available
#[cfg(not(feature = "minimal"))]
pub use hybrid::{
    HybridIndex, EmbeddingConfig, SemanticSearch, RagEngine, RagConfig,
    VectorCypherParser, GraphNeuralEngine, GnnConfig,
};

// Re-export distributed types when feature is enabled
#[cfg(feature = "distributed")]
pub use distributed::{
    Coordinator, Federation, GossipMembership, GraphReplication, GraphShard, RpcClient, RpcServer,
    ShardCoordinator, ShardStrategy,
};

#[cfg(test)]
mod tests {
    #[test]
    fn test_placeholder() {
        // Placeholder test to allow compilation
        assert!(true);
    }
}
