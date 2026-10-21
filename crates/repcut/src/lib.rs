// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH. Modifications to the original file.
//! Partition a staged AIG into GPU-mappable clusters via the RepCut algorithm.
//!
//! Each cluster becomes one partition that runs on one Boolean processor
//! (256 CUDA threads). The algorithm — RepCut [[Wang & Beamer, ASPLOS
//! 2023]][paper] — starts from sink nodes, crawls upward to identify logic
//! cones, clusters nodes by cone membership, builds a hypergraph from the
//! clustering, and invokes mt-kahypar to partition it. Logic shared across
//! partition boundaries is *replicated* rather than communicated, trading
//! duplicated work for zero inter-PE dependency.
//!
//! Upstream: [`staging`] produces the [`StagedAIG`] inputs.
//! Downstream: [`flatten`] consumes the partition assignment and lowers it
//! into a VLIW bitstream.
//!
//! # Key types
//!
//! - [`RCHyperGraph`] — the hypergraph handed to mt-kahypar, plus the cone
//!   clustering it was built from.
//!
//! # Notes
//!
//! Source nodes representing register reads are not partitioned — they are
//! broadcast to every partition that observes them, alongside primary inputs.
//!
//! [paper]: https://dl.acm.org/doi/10.1145/3582016.3582034
//! [`staging`]: ../staging/
//! [`flatten`]: ../flatten/

use std::fmt;

use aig::AIG;
use staging::StagedAIG;

use crate::{
    cones::{ConeClusters, HGAdjacency},
    weights::EndpointWeights,
};

mod cones;
mod weights;

pub(crate) fn vec_try_into<T, U>(v: impl IntoIterator<Item = T>) -> Vec<U>
where
    T: TryInto<U>,
    T::Error: std::fmt::Debug,
{
    v.into_iter().map(|x| x.try_into().unwrap()).collect()
}

pub struct RCHyperGraph {
    num_vertices: usize,
    clusters: ConeClusters,
    endpoint_weights: Vec<u64>,
}

// TODO this should go into a writer?
impl fmt::Display for RCHyperGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{} {} 11",
            self.clusters.len(),
            self.endpoint_weights.len()
        )?;
        for (v, edgend) in self.clusters.to_edges() {
            write!(f, "{}", v)?;
            for nd in edgend {
                write!(f, " {}", nd + 1)?;
            }
            writeln!(f)?;
        }
        for w in &self.endpoint_weights {
            writeln!(f, "{}", w)?;
        }
        Ok(())
    }
}

impl RCHyperGraph {
    /// Num vertices in the graph
    pub fn num_vertices(&self) -> usize {
        self.num_vertices
    }

    pub fn from_staged_aig(aig: &AIG, staged: &StagedAIG) -> Self {
        let num_endpoint_groups = staged.num_endpoint_groups();

        let _span = tracing::info_span!("repcut endpoint process").entered();
        let clusters =
            ConeClusters::from_staged_aig(aig, staged, num_endpoint_groups);

        let endpoint_weights =
            EndpointWeights::from_staged_aig(aig, staged, num_endpoint_groups);

        Self {
            num_vertices: num_endpoint_groups,
            clusters,
            endpoint_weights,
        }
    }

    /// Run mt-kahypar to partition this hypergraph.
    /// Returns partition ids for each node
    pub fn partition(&self, num_parts: usize) -> Vec<usize> {
        // Handle the special case where num_parts = 1 (mt-kahypar requires k >=
        // 2)
        if num_parts == 1 {
            return vec![0; self.num_vertices];
        }

        let params = mt_kahypar::PartitioningParams {
            k: num_parts.try_into().unwrap(),
            epsilon: 0.2,
            objective: mt_kahypar::Objective::Soed,
        };

        let init = mt_kahypar::Init::with_default_threads();
        let ctx = mt_kahypar::Context::builder(&init)
            .preset(mt_kahypar::Preset::Deterministic)
            .params(params)
            .verbose(true)
            .build()
            .unwrap();

        let adjacency = HGAdjacency::from(&self.clusters);
        let vertex_weights: Vec<i32> =
            vec_try_into(self.endpoint_weights.iter().copied());
        let hg = mt_kahypar::Hypergraph::from_adjacency(
            &ctx,
            self.num_vertices,
            &adjacency.indices,
            &adjacency.edges,
            Some(&adjacency.edge_weights),
            Some(&vertex_weights),
        )
        .unwrap();

        let parts = hg.partition().unwrap();
        vec_try_into(parts.extract_partition())
    }
}
