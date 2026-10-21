// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH. Modifications to the original file.
use std::{collections::HashMap, sync::Arc};

use aig::AIG;
use cachedhash::CachedHash;
use indexmap::{IndexMap, IndexSet};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use staging::StagedAIG;

const REPCUT_HYPERGRAPH_EDGE_SIZE_LIMIT: usize = 1000;
const REPCUT_BITSET_BLOCK_SIZE: usize = 4096;

/// Compute endpoint index from segment, u64 block index, and bit position
macro_rules! endpoint_index {
    ($segment_i:expr, $block_i:expr, $k:expr) => {
        $segment_i * REPCUT_BITSET_BLOCK_SIZE + $block_i * 64 + $k
    };
}

/// Fixed-size bitset representing which endpoints (within a block of
/// REPCUT_BITSET_BLOCK_SIZE) a node can reach, e.g. bit b in `bs_set[k]` is set
/// if the node reaches endpoint `block * 4096 + k * 64 + b`.
#[derive(Hash, PartialEq, Eq, Debug)]
struct EndpointSetSegment {
    bs_set: [u64; REPCUT_BITSET_BLOCK_SIZE / 64],
}

// Arc and CachedHash are memory optimisations (sharing across nodes in
// par_iter)
type CachedEndpointSetSegment = Arc<CachedHash<EndpointSetSegment>>;

impl Default for EndpointSetSegment {
    fn default() -> Self {
        EndpointSetSegment {
            bs_set: [0; REPCUT_BITSET_BLOCK_SIZE / 64],
        }
    }
}

/// Vector of optional segments, one per block. Represents the full reachability
/// of a node across all endpoints
#[derive(Hash, PartialEq, Eq, Debug)]
struct EndpointSet(Vec<Option<CachedEndpointSetSegment>>);

impl EndpointSet {
    /// count number of set bits
    fn popcount(&self) -> usize {
        self.0
            .iter()
            .map(|o| match o {
                Some(ess) => {
                    ess.bs_set.iter().map(|u| u.count_ones()).sum::<u32>()
                        as usize
                }
                None => 0,
            })
            .sum()
    }
}

pub(crate) struct HGAdjacency {
    pub(crate) indices: Vec<usize>,
    pub(crate) edges: Vec<usize>,
    pub(crate) edge_weights: Vec<i32>,
}

/// An IndexMap from EndpointSet -> count of nodes with that reachability
/// pattern
pub(crate) struct ConeClusters(IndexMap<CachedHash<EndpointSet>, usize>);

/// Construct an hypergraph adjacency from ConeClusters to be used with
/// [`mt_kahypar::Hypergraph::from_adjacency`]
impl From<&ConeClusters> for HGAdjacency {
    fn from(clusters: &ConeClusters) -> Self {
        let edges = clusters.to_edges();

        let indices: Vec<_> = std::iter::once(0)
            .chain(edges.iter().scan(0, |acc, (_, edgend)| {
                *acc += edgend.len();
                Some(*acc)
            }))
            .collect();

        let hg_edges: Vec<_> = edges
            .iter()
            .flat_map(|(_, edgend)| edgend.iter().copied())
            .collect();

        let edge_weights: Vec<i32> =
            crate::vec_try_into(edges.iter().map(|&(v, _)| v));

        HGAdjacency {
            indices,
            edges: hg_edges,
            edge_weights,
        }
    }
}

impl ConeClusters {
    /// For each node in the AIG, it stores a bitset indicating which endpoints
    /// that node can reach (i.e., which endpoints are in the node's "cone
    /// of influence").
    ///
    /// The algorithm:
    /// 1. For each block, it first marks direct inputs to endpoints in
    ///    ess_inputs
    /// 2. Then traverses in reverse topological order, propagating reachability
    ///    backwards - if node j reaches endpoint E, and node i fans out to j,
    ///    then node i also reaches endpoint E
    /// 3. The bitset bs_set[k] has bit b set if the node reaches endpoint
    ///    (block * 4096 + k * 64 + b)
    ///
    /// These are then combined into EndpointSet (one segment per block) for
    /// each node. Nodes with identical EndpointSet (same reachability
    /// pattern to endpoints) get clustered together.
    pub(crate) fn from_staged_aig(
        aig: &AIG,
        staged: &StagedAIG,
        num_endpoint_groups: usize,
    ) -> Self {
        // endpoints are partitioned into blocks of size
        // REPCUT_BITSET_BLOCK_SIZE
        let num_blocks = num_endpoint_groups.div_ceil(REPCUT_BITSET_BLOCK_SIZE);

        // TODO type
        let segments_blockid_nodeid =
            Self::segments(aig, staged, num_endpoint_groups, num_blocks);

        // convert/collect EndpointSetSegment into EndpointSet with number of
        // occurences
        let mut clusters = IndexMap::<_, usize>::new();
        #[allow(clippy::needless_range_loop)]
        for i in 1..aig.num_aigpins {
            let es = CachedHash::new(EndpointSet(
                (0..num_blocks)
                    .map(|k| segments_blockid_nodeid[k][i].clone())
                    .collect(),
            ));
            if es.popcount() >= 2 {
                *clusters.entry(es).or_default() += 1;
            }
        }

        Self(clusters)
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Make an edge list not exceeding REPCUT_HYPERGRAPH_EDGE_SIZE_LIMIT
    /// (reservoir sampling).
    ///
    /// Returns list of (weight, node indices).
    pub fn to_edges(&self) -> Vec<(usize, Vec<usize>)> {
        self.0
            .par_iter()
            .enumerate()
            .map(|(i, (epts, v))| {
                // deterministic rng for reservoir sampling
                let mut rng = ChaCha20Rng::seed_from_u64(8026727 + i as u64);
                let mut edge_nd = Vec::<usize>::new();
                let mut num_prev_nodes = 0;
                for (segment_i, seg) in epts
                    .0
                    .iter()
                    .enumerate()
                    .filter_map(|(i, opt)| opt.as_ref().map(|seg| (i, seg)))
                {
                    for block_i in 0..REPCUT_BITSET_BLOCK_SIZE / 64 {
                        if seg.bs_set[block_i] == 0 {
                            continue;
                        }

                        for k in 0..64 {
                            if (seg.bs_set[block_i] >> k & 1) != 0 {
                                let nd = endpoint_index!(segment_i, block_i, k);

                                // if we exceed the edge size limit use
                                // reservoir sampling to keep a representative
                                // subset
                                if edge_nd.len()
                                    < REPCUT_HYPERGRAPH_EDGE_SIZE_LIMIT
                                {
                                    edge_nd.push(nd);
                                } else if rng.gen_range(0..num_prev_nodes)
                                    < REPCUT_HYPERGRAPH_EDGE_SIZE_LIMIT
                                {
                                    edge_nd[rng.gen_range(
                                        0..REPCUT_HYPERGRAPH_EDGE_SIZE_LIMIT,
                                    )] = nd;
                                }
                                num_prev_nodes += 1;
                            }
                        }
                    }
                }
                (*v, edge_nd)
            })
            .collect()
    }

    /// returns a map which endpoints in `block` are reachable from `node`
    fn segments(
        aig: &AIG,
        staged: &StagedAIG,
        num_endpoint_groups: usize,
        num_blocks: usize,
    ) -> Vec<Vec<Option<CachedEndpointSetSegment>>> {
        let mut segments =
            vec![Vec::<Option<CachedEndpointSetSegment>>::new(); num_blocks];
        segments.par_iter_mut().enumerate().for_each(
            |(i_block, node_to_segment)| {
                *node_to_segment = vec![None; aig.num_aigpins + 1];

                let block_start = i_block * REPCUT_BITSET_BLOCK_SIZE;
                let block_end = num_endpoint_groups
                    .min(block_start + REPCUT_BITSET_BLOCK_SIZE);
                let block_range = block_start..block_end;

                // map node to initial EndpointSetSegement for nodes that
                // directly feed endpoints
                let mut ess_inputs = {
                    let mut ess_inputs: HashMap<usize, EndpointSetSegment> =
                        HashMap::new();
                    for (endpt_i, i) in
                        block_range.clone().flat_map(|endpt_i| {
                            staged
                                .get_endpoint_group(aig, endpt_i)
                                .inputs()
                                .map(move |i| (endpt_i, i))
                        })
                    {
                        let idx_offset = endpt_i - block_start;
                        ess_inputs.entry(i).or_default().bs_set
                            [idx_offset / 64] |= 1 << (idx_offset % 64);
                    }
                    ess_inputs
                };

                // topologically ordered nodes of the current blocks endpoints
                let order_blk = {
                    let endpoint_pins: Vec<_> = block_range
                        .flat_map(|endpt_i| {
                            staged.get_endpoint_group(aig, endpt_i).inputs()
                        })
                        .collect();

                    aig.topo_traverse_generic(
                        Some(&endpoint_pins),
                        staged.primary_inputs.as_ref(),
                    )
                };

                // deduplicate identical segments
                let mut unique_segs =
                    IndexSet::<CachedEndpointSetSegment>::new();
                // traverse in reverse topological order
                for &i in order_blk.iter().rev() {
                    let mut ess = ess_inputs.remove(&i).unwrap_or_default();
                    // check reachable from all fanouts
                    for &j in &aig.fanouts
                        [aig.fanouts_start[i]..aig.fanouts_start[i + 1]]
                    {
                        if let Some(vj) = &mut node_to_segment[j] {
                            for bs_k in 0..REPCUT_BITSET_BLOCK_SIZE / 64 {
                                ess.bs_set[bs_k] |= vj.bs_set[bs_k];
                            }
                        }
                    }
                    let ess = Arc::new(CachedHash::new(ess));
                    let (idx, _) = unique_segs.insert_full(ess);
                    node_to_segment[i] =
                        Some(unique_segs.get_index(idx).unwrap().clone());
                }
            },
        );

        segments
    }
}
