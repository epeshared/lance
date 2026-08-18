// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Builds an HNSW graph from an exact neighbour table instead of inserting the
//! nodes one at a time.
//!
//! The incremental builder pays a beam search per insertion to find that node's
//! candidate neighbours. This path instead computes every node's exact top
//! `ef_construction` neighbours once for the whole partition (see
//! [`super::exact_knn`], which runs that as a GEMM on the AMX-FP16 tile kernel)
//! and selects each node's edges straight from that table.
//!
//! What comes out is an ordinary [`HNSW`]: the same node levels off the same
//! seed, the same [`select_neighbors_heuristic_owned`] pruning, the same
//! `GraphBuilderNode` layout. Nothing downstream — serialization, load, search —
//! can tell the two apart, which is why this is a build strategy rather than an
//! index type.
//!
//! ## Two deliberate differences from incremental insertion
//!
//! **Every level gets its own exact table.** Level `l` only connects the nodes
//! assigned to it, roughly `n / m^l` of them, so the partition-wide top-k is
//! useless there: at `m = 20` the level-1 members are 5% of the partition and a
//! global top-150 can easily contain none of them. Each upper level therefore
//! gathers its members' vectors into a contiguous array and runs its own
//! [`exact_knn_topk`] over just those. The cost is geometric in the level, so
//! level 1 adds ~0.5% to the level-0 matrix and the rest is noise.
//!
//! **Back edges are added unconditionally.** Incremental insertion only offers
//! the reverse edge to a chosen neighbour and lets it refuse (the `cutoff`
//! test). Here `i -> j` and `j -> i` are unioned outright and any node pushed
//! past its degree cap is re-pruned. Offline measurement attributes most of the
//! low-`ef` recall difference between the two paths to exactly this, and next to
//! none of it to the candidates being exact.
//!
//! ## Fallback
//!
//! [`build_from_exact_knn`] answers `Ok(None)` for anything it cannot serve —
//! wrong element type, wrong metric, no AMX, partition over the configured size.
//! The caller then builds incrementally. The only errors it raises are real
//! ones.

use std::sync::Arc;

use half::f16;
use lance_core::{Error, Result};
use lance_linalg::distance::DistanceType;
use lance_linalg::distance::dot_f16::amx_fp16_available;
use rayon::prelude::*;

use super::builder::{HNSW, HnswBuildParams, assign_node_levels, entry_point_of};
use super::exact_knn::{ExactKnn, exact_knn_topk};
use super::select_neighbors_heuristic_owned;
use crate::vector::flat::storage::FlatFloatStorage;
use crate::vector::graph::builder::GraphBuilderNode;
use crate::vector::graph::{OrderedFloat, OrderedNode};
use crate::vector::storage::VectorStore;

/// Placeholder in the global-id -> level-index map for a node that is not on the
/// level. `u32::MAX` cannot collide with a real index: a partition that large is
/// rejected by [`exact_knn_topk`] long before it gets here.
const NOT_ON_LEVEL: u32 = u32::MAX;

/// Builds `storage`'s graph from exact neighbour tables, or answers `Ok(None)`
/// if this partition is not one the path can serve, in which case the caller
/// must fall back to incremental insertion.
///
/// Runs the neighbour tables on the calling thread — the intended deployment
/// gives each partition a thread of its own — and parallelises only the
/// per-node edge selection over the current rayon pool.
pub fn build_from_exact_knn(
    storage: &impl VectorStore,
    params: &HnswBuildParams,
) -> Result<Option<HNSW>> {
    let Some((vectors, dim)) = eligible_vectors(storage, params) else {
        return Ok(None);
    };

    match build(storage, params, vectors, dim) {
        Ok(hnsw) => Ok(Some(hnsw)),
        // `eligible_vectors` already checked everything this path knows how to
        // check, so a kernel that still declines is something it does not know
        // about; say so once and build the graph the other way rather than
        // failing the index.
        Err(error @ Error::NotSupported { .. }) => {
            log::warn!(
                "HNSW exact-knn construction unavailable for a {} vector partition, \
                 falling back to incremental insertion: {error}",
                storage.len()
            );
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

/// The partition's `f16` vectors and dimension, if every precondition of the
/// exact path holds. `None` means "build incrementally", never an error: each
/// condition below is a legitimate partition this path simply does not cover.
fn eligible_vectors<'a>(
    storage: &'a impl VectorStore,
    params: &HnswBuildParams,
) -> Option<(&'a [f16], usize)> {
    if !params.use_exact_knn_construction {
        return None;
    }
    // `assign_node_levels` draws against `max_level - 1`, and the level loops
    // below have nothing to iterate over at `max_level = 0`.
    if params.max_level == 0 || storage.is_empty() {
        return None;
    }
    if storage.len() > params.exact_knn_max_partition_size {
        log::debug!(
            "HNSW exact-knn construction skipped: partition of {} vectors exceeds the \
             configured maximum of {}",
            storage.len(),
            params.exact_knn_max_partition_size
        );
        return None;
    }
    // The neighbour table is only implemented for `Dot`, and only on the
    // AMX-FP16 GEMM, which needs `f16` and a host that grants tile registers.
    if storage.distance_type() != DistanceType::Dot || !amx_fp16_available() {
        return None;
    }
    let flat = storage.as_any().downcast_ref::<FlatFloatStorage>()?;
    flat.f16_vectors()
}

fn build(
    storage: &impl VectorStore,
    params: &HnswBuildParams,
    vectors: &[f16],
    dim: usize,
) -> Result<HNSW> {
    let len = storage.len();
    let levels = assign_node_levels(len, params);
    let max_level = params.max_level as usize;

    log::debug!(
        "Building HNSW graph from exact neighbours: num={len}, dim={dim}, \
         max_levels={max_level}, m={}, ef_construction={}",
        params.m,
        params.ef_construction,
    );

    // Level 0 holds every node, so it is its own member list and needs no gather.
    let mut ranked: Vec<Vec<Vec<OrderedNode>>> = levels
        .iter()
        .map(|&node_levels| vec![Vec::new(); node_levels])
        .collect();

    let members: Vec<u32> = (0..len as u32).collect();
    let knn = exact_knn_topk(vectors, len, dim, params.ef_construction, DistanceType::Dot)?;
    scatter_level(
        &mut ranked,
        0,
        &members,
        level_edges(storage, &members, &knn, 2 * params.m)?,
    );

    // Upper levels: only the nodes assigned to the level may be neighbours on
    // it, so each one needs a neighbour table over just its own members.
    let mut level_vectors = Vec::new();
    for level in 1..max_level {
        let members: Vec<u32> = (0..len as u32)
            .filter(|&id| levels[id as usize] > level)
            .collect();
        if members.len() < 2 {
            continue;
        }
        level_vectors.clear();
        level_vectors.reserve(members.len() * dim);
        for &id in &members {
            let start = id as usize * dim;
            level_vectors.extend_from_slice(&vectors[start..start + dim]);
        }
        let knn = exact_knn_topk(
            &level_vectors,
            members.len(),
            dim,
            params.ef_construction,
            DistanceType::Dot,
        )?;
        scatter_level(
            &mut ranked,
            level,
            &members,
            level_edges(storage, &members, &knn, params.m)?,
        );
    }

    // `level_count[l]` is how many nodes reach level `l` -- what incremental
    // insertion accumulates as it walks each node's levels down from its own.
    let level_count = (0..max_level)
        .map(|level| {
            levels
                .iter()
                .filter(|&&node_levels| node_levels > level)
                .count()
        })
        .collect();
    let entry_point = entry_point_of(&levels);

    let nodes = ranked.into_iter().map(into_builder_node).collect();
    Ok(HNSW::from_parts(
        params.clone(),
        nodes,
        level_count,
        entry_point,
    ))
}

/// One level's adjacency, indexed by position in `members` and holding global
/// node ids: each member's neighbours selected from its exact candidates, then
/// merged with the edges that chose it.
///
/// `knn` must be the neighbour table of `members`' vectors in `members` order,
/// so its ids are positions in `members` rather than node ids.
fn level_edges(
    storage: &impl VectorStore,
    members: &[u32],
    knn: &ExactKnn,
    m_max: usize,
) -> Result<Vec<Vec<OrderedNode>>> {
    // HNSW paper Algorithm 4, over exact candidates instead of beam-searched
    // ones. `select_neighbors_heuristic_owned` wants global ids because it asks
    // `storage` for candidate-to-candidate distances.
    let selected: Vec<Vec<OrderedNode>> = (0..members.len())
        .into_par_iter()
        .map(|member| {
            let (ids, dists) = knn.neighbours(member);
            let candidates = ids
                .iter()
                .zip(dists)
                .map(|(&id, &dist)| OrderedNode::new(members[id as usize], OrderedFloat(dist)))
                .collect();
            select_neighbors_heuristic_owned(storage, candidates, m_max)
        })
        .collect();

    merge_bidirectional(storage, members, selected, m_max)
}

/// Adds the reverse of every edge, unconditionally, and re-prunes whoever that
/// pushes past `m_max`.
///
/// Unlike incremental insertion, where a node may refuse a back edge that is
/// worse than its current worst neighbour, this makes the level's edge set
/// symmetric before pruning. Offline measurement puts most of the low-`ef`
/// recall difference between the two paths here.
fn merge_bidirectional(
    storage: &impl VectorStore,
    members: &[u32],
    mut adjacency: Vec<Vec<OrderedNode>>,
    m_max: usize,
) -> Result<Vec<Vec<OrderedNode>>> {
    // Positions in `members`, so a neighbour's global id can be turned back into
    // the row of `adjacency` that has to gain the reverse edge.
    let mut member_index = vec![NOT_ON_LEVEL; storage.len()];
    for (index, &id) in members.iter().enumerate() {
        member_index[id as usize] = index as u32;
    }

    let mut reverse: Vec<Vec<OrderedNode>> = vec![Vec::new(); adjacency.len()];
    for (index, edges) in adjacency.iter().enumerate() {
        let source = members[index];
        for edge in edges {
            let target = member_index
                .get(edge.id as usize)
                .copied()
                .filter(|&target| target != NOT_ON_LEVEL)
                .ok_or_else(|| {
                    Error::internal(format!(
                        "HNSW exact-knn construction picked node {} as a neighbour of node \
                         {source}, but it is not on this level",
                        edge.id
                    ))
                })?;
            reverse[target as usize].push(OrderedNode::new(source, edge.dist));
        }
    }

    adjacency
        .par_iter_mut()
        .zip(reverse)
        .for_each(|(edges, reverse)| {
            edges.extend(reverse);
            // An edge chosen from both ends arrives twice. Both copies carry the
            // same distance, but `OrderedNode` orders on distance alone, so
            // duplicates are only guaranteed adjacent when sorted by id.
            edges.sort_unstable_by_key(|edge| edge.id);
            edges.dedup_by_key(|edge| edge.id);
            if edges.len() > m_max {
                let candidates = std::mem::take(edges);
                *edges = select_neighbors_heuristic_owned(storage, candidates, m_max);
            }
            edges.sort_unstable();
        });

    Ok(adjacency)
}

/// Files one level's adjacency into the per-node lists, translating positions in
/// `members` back to node ids.
fn scatter_level(
    ranked: &mut [Vec<Vec<OrderedNode>>],
    level: usize,
    members: &[u32],
    edges: Vec<Vec<OrderedNode>>,
) {
    for (&id, node_edges) in members.iter().zip(edges) {
        ranked[id as usize][level] = node_edges;
    }
}

/// Turns one node's per-level ranked edges into the node the graph stores,
/// mirroring level 0 into the bottom-level list search reads.
fn into_builder_node(ranked: Vec<Vec<OrderedNode>>) -> GraphBuilderNode {
    let level_neighbors: Vec<Arc<Vec<u32>>> = ranked
        .iter()
        .map(|edges| Arc::new(edges.iter().map(|edge| edge.id).collect()))
        .collect();
    // Every node is on level 0 (`assign_node_levels` never returns 0 levels), so
    // the empty list is a value rather than a fallback.
    let bottom_neighbors = level_neighbors
        .first()
        .cloned()
        .unwrap_or_else(|| Arc::new(Vec::new()));
    GraphBuilderNode::from_parts(level_neighbors, ranked, bottom_neighbors)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashSet;

    use arrow_array::{ArrayRef, FixedSizeListArray, Float16Array, Float32Array};
    use lance_arrow::FixedSizeListArrayExt;

    use crate::vector::graph::Graph;
    use crate::vector::hnsw::builder::{HnswQueryParams, ImmutableHnswBottomView};
    use crate::vector::storage::DistCalculator;
    use crate::vector::v3::subindex::IvfSubIndex;

    /// The neighbour table only exists where the tile kernel was compiled in and
    /// the host grants tile registers, so tests that need a real graph return
    /// early elsewhere — the same convention `exact_knn`'s tests use.
    fn amx_ready() -> bool {
        amx_fp16_available()
    }

    /// `n` deterministic pseudo-random unit vectors, row-major.
    fn random_unit_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / 16_777_216.0 - 0.5
        };
        let mut out = Vec::with_capacity(n * dim);
        for _ in 0..n {
            let mut v: Vec<f32> = (0..dim).map(|_| next()).collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            v.iter_mut().for_each(|x| *x /= norm);
            out.extend(v);
        }
        out
    }

    fn f16_storage(
        n: usize,
        dim: usize,
        seed: u64,
        distance_type: DistanceType,
    ) -> FlatFloatStorage {
        let values = Float16Array::from_iter_values(
            random_unit_vectors(n, dim, seed)
                .into_iter()
                .map(f16::from_f32),
        );
        let vectors = FixedSizeListArray::try_new_from_values(values, dim as i32).unwrap();
        FlatFloatStorage::new(vectors, distance_type)
    }

    fn test_params() -> HnswBuildParams {
        HnswBuildParams::default()
            .num_edges(8)
            .ef_construction(32)
            .max_level(4)
            .with_exact_knn_construction(true)
    }

    #[test]
    fn test_declines_what_it_cannot_serve() {
        let params = test_params();
        let dim = 64;

        // Off by default.
        let storage = f16_storage(64, dim, 1, DistanceType::Dot);
        let off = HnswBuildParams::default();
        assert!(build_from_exact_knn(&storage, &off).unwrap().is_none());

        // Wrong metric.
        let l2 = f16_storage(64, dim, 1, DistanceType::L2);
        assert!(build_from_exact_knn(&l2, &params).unwrap().is_none());

        // Wrong element type.
        let f32_values = Float32Array::from(random_unit_vectors(64, dim, 1));
        let f32_vectors = FixedSizeListArray::try_new_from_values(f32_values, dim as i32).unwrap();
        let f32_storage = FlatFloatStorage::new(f32_vectors, DistanceType::Dot);
        assert!(
            build_from_exact_knn(&f32_storage, &params)
                .unwrap()
                .is_none()
        );

        // Partition over the configured ceiling.
        let capped = params.clone().with_exact_knn_max_partition_size(16);
        assert!(build_from_exact_knn(&storage, &capped).unwrap().is_none());

        // And it does serve the case all of those differ from — but only where
        // the kernel exists.
        assert_eq!(
            build_from_exact_knn(&storage, &params).unwrap().is_some(),
            amx_ready()
        );
    }

    /// Everything the search path assumes about a graph's shape, checked on the
    /// graph this module produces.
    #[test]
    fn test_graph_structure_invariants() {
        if !amx_ready() {
            return;
        }
        // Past 1024 rows and 256 columns the neighbour table walks several
        // blocks of the distance matrix, including a short trailing one.
        let (n, dim) = (2500, 64);
        let params = test_params();
        let storage = f16_storage(n, dim, 3, DistanceType::Dot);
        let hnsw = build_from_exact_knn(&storage, &params).unwrap().unwrap();
        let nodes = hnsw.nodes().unwrap();

        assert_eq!(nodes.len(), n);
        let levels = assign_node_levels(n, &params);
        for (id, node) in nodes.iter().enumerate() {
            assert_eq!(
                node.level_neighbors.len(),
                levels[id],
                "node {id} is on the wrong number of levels"
            );
            assert_eq!(
                node.bottom_neighbors.as_ref(),
                node.level_neighbors[0].as_ref(),
                "node {id}'s bottom neighbours drifted from level 0"
            );
        }

        for level in 0..params.max_level as usize {
            let m_max = if level == 0 { 2 * params.m } else { params.m };
            let on_level: HashSet<u32> = (0..n as u32)
                .filter(|&id| levels[id as usize] > level)
                .collect();
            assert_eq!(
                hnsw.num_nodes(level),
                on_level.len(),
                "level {level}'s node count disagrees with the level assignment"
            );

            for &id in &on_level {
                let neighbors = nodes[id as usize].level_neighbors[level].as_ref();
                assert!(
                    neighbors.len() <= m_max,
                    "node {id} has {} neighbours on level {level}, over the cap of {m_max}",
                    neighbors.len()
                );
                assert!(
                    !neighbors.contains(&id),
                    "node {id} has a self loop on level {level}"
                );
                let unique: HashSet<u32> = neighbors.iter().copied().collect();
                assert_eq!(
                    unique.len(),
                    neighbors.len(),
                    "node {id} has a duplicate neighbour on level {level}"
                );
                for neighbor in neighbors {
                    assert!(
                        on_level.contains(neighbor),
                        "node {id}'s level-{level} neighbour {neighbor} is not on that level"
                    );
                }
                assert_eq!(
                    nodes[id as usize].level_neighbors_ranked[level].len(),
                    neighbors.len(),
                    "node {id}'s level-{level} ranked list disagrees with its neighbour list"
                );
            }
        }

        // Algorithm 4 always takes the closest candidate, and on level 0 the
        // candidates are the exact top-`ef_construction`, so every node's first
        // neighbour has to be its true nearest neighbour. Nothing else in this
        // file fails as loudly when a distance is inverted or an id is mapped
        // through the wrong level.
        for id in 0..n as u32 {
            let calc = storage.dist_calculator_from_id(id);
            let nearest = (0..n as u32)
                .filter(|&other| other != id)
                .map(|other| OrderedFloat(calc.distance(other)))
                .min()
                .unwrap();
            let chosen = nodes[id as usize].level_neighbors_ranked[0][0].dist;
            assert!(
                (chosen.0 - nearest.0).abs() < 1e-3,
                "node {id}'s closest level-0 edge is at {chosen:?}, but its nearest \
                 neighbour is at {nearest:?}"
            );
        }

        // Search only works if the entry point reaches the rest of each level:
        // level 0 for the beam, the upper ones for the greedy descent. This is
        // also what catches an upper level built from the partition-wide
        // neighbour table instead of its own — most of its members would end up
        // with no in-level candidates at all, and the level would shatter.
        let entry_point = entry_point_of(&levels);
        for level in 0..params.max_level as usize {
            let members: Vec<u32> = (0..n as u32)
                .filter(|&id| levels[id as usize] > level)
                .collect();
            // The entry point is the tallest node, so it is on every level that
            // has any member at all; above that there is nothing to reach.
            if members.is_empty() {
                continue;
            }
            let mut seen = HashSet::from([entry_point]);
            let mut queue = vec![entry_point];
            while let Some(id) = queue.pop() {
                for &neighbor in nodes[id as usize].level_neighbors[level].iter() {
                    if seen.insert(neighbor) {
                        queue.push(neighbor);
                    }
                }
            }
            assert_eq!(
                seen.len(),
                members.len(),
                "level {level} is not fully reachable from the entry point: \
                 {} of {} members",
                seen.len(),
                members.len()
            );
        }
    }

    /// Nothing in this path draws at build time, so two builds of the same
    /// partition must agree edge for edge. (Incremental insertion does not: its
    /// nodes race each other.)
    #[test]
    fn test_build_is_deterministic() {
        if !amx_ready() {
            return;
        }
        let params = test_params();
        let storage = f16_storage(400, 64, 5, DistanceType::Dot);
        let first = build_from_exact_knn(&storage, &params).unwrap().unwrap();
        let second = build_from_exact_knn(&storage, &params).unwrap().unwrap();

        let (first, second) = (first.nodes().unwrap(), second.nodes().unwrap());
        assert_eq!(first.len(), second.len());
        for (id, (a, b)) in first.iter().zip(second.iter()).enumerate() {
            assert_eq!(
                a.level_neighbors, b.level_neighbors,
                "node {id} differs between builds"
            );
        }
    }

    /// Recall of the two construction paths, measured against brute force on the
    /// same storage and the same queries.
    ///
    /// The graphs are different by design, so this is not a parity check; it is
    /// the only thing that catches a silently wrong build (padding slots leaking
    /// into candidate lists, upper-level ids not mapped back, a flipped distance
    /// sign), all of which produce a graph that searches fine and just answers
    /// worse.
    #[test]
    fn test_recall_is_not_worse_than_incremental() {
        if !amx_ready() {
            return;
        }
        let (n, dim, k) = (1200, 64, 10);
        let params = test_params();
        let storage = f16_storage(n, dim, 7, DistanceType::Dot);

        let exact = build_from_exact_knn(&storage, &params).unwrap().unwrap();
        let incremental =
            HNSW::index_vectors(&storage, params.with_exact_knn_construction(false)).unwrap();

        let queries = random_unit_vectors(50, dim, 9);
        for &ef in &[16usize, 32, 128] {
            let mut exact_hits = 0;
            let mut incremental_hits = 0;
            let mut total = 0;
            for query in queries.chunks_exact(dim) {
                let query: ArrayRef = Arc::new(Float16Array::from_iter_values(
                    query.iter().copied().map(f16::from_f32),
                ));
                let truth = brute_force(&storage, query.clone(), k);
                total += truth.len();
                exact_hits += hits(&exact, &storage, query.clone(), k, ef, &truth);
                incremental_hits += hits(&incremental, &storage, query, k, ef, &truth);
            }
            let exact_recall = exact_hits as f32 / total as f32;
            let incremental_recall = incremental_hits as f32 / total as f32;
            // The margin absorbs the incremental path's run-to-run spread --
            // its insertions race, so its recall moves by a few points between
            // runs while this path's is fixed. Measured over five runs of this
            // test: 0.668 / 0.836 / 0.998 here against 0.574-0.592 /
            // 0.732-0.762 / 0.956-0.980 there, so the margin never decides it.
            assert!(
                exact_recall >= incremental_recall - 0.03,
                "ef={ef}: exact-knn recall {exact_recall} trails incremental {incremental_recall}"
            );
            // With a beam that wide on a partition this small, anything short of
            // near-perfect means the graph itself is wrong.
            if ef >= 128 {
                assert!(
                    exact_recall >= 0.95,
                    "ef={ef}: exact-knn recall {exact_recall} is too low to be a working graph"
                );
            }
        }
    }

    fn brute_force(storage: &FlatFloatStorage, query: ArrayRef, k: usize) -> Vec<u64> {
        let calc = storage.dist_calculator(query, 0.0);
        let mut scored: Vec<(OrderedFloat, u32)> = (0..storage.len() as u32)
            .map(|id| (OrderedFloat(calc.distance(id)), id))
            .collect();
        scored.sort_unstable();
        scored
            .into_iter()
            .take(k)
            .map(|(_, id)| storage.row_id(id))
            .collect()
    }

    fn hits(
        hnsw: &HNSW,
        storage: &FlatFloatStorage,
        query: ArrayRef,
        k: usize,
        ef: usize,
        truth: &[u64],
    ) -> usize {
        let params = HnswQueryParams {
            ef,
            lower_bound: None,
            upper_bound: None,
            dist_q_c: 0.0,
            use_acorn: false,
        };
        let found: HashSet<u64> = hnsw
            .search_basic(query, k, &params, None, storage)
            .unwrap()
            .into_iter()
            .map(|node| storage.row_id(node.id))
            .collect();
        truth.iter().filter(|id| found.contains(id)).count()
    }

    /// The graph has to survive the round trip the index writer puts it through.
    #[test]
    fn test_built_graph_round_trips_through_batch() {
        if !amx_ready() {
            return;
        }
        let (n, dim) = (300, 64);
        let params = test_params();
        let storage = f16_storage(n, dim, 11, DistanceType::Dot);
        let hnsw = build_from_exact_knn(&storage, &params).unwrap().unwrap();
        let loaded = HNSW::load(hnsw.to_batch().unwrap()).unwrap();

        assert_eq!(loaded.len(), hnsw.len());
        for level in 0..params.max_level as usize {
            assert_eq!(loaded.num_nodes(level), hnsw.num_nodes(level));
        }

        let query: ArrayRef = Arc::new(Float16Array::from_iter_values(
            random_unit_vectors(1, dim, 13)
                .into_iter()
                .map(f16::from_f32),
        ));
        let query_params = HnswQueryParams {
            ef: 64,
            lower_bound: None,
            upper_bound: None,
            dist_q_c: 0.0,
            use_acorn: false,
        };
        let before = hnsw
            .search_basic(query.clone(), 10, &query_params, None, &storage)
            .unwrap();
        let after = loaded
            .search_basic(query, 10, &query_params, None, &storage)
            .unwrap();
        assert_eq!(
            before.iter().map(|n| n.id).collect::<Vec<_>>(),
            after.iter().map(|n| n.id).collect::<Vec<_>>(),
            "the reloaded graph searches differently from the built one"
        );
    }

    /// A partition with fewer nodes than `m` still has to come out as a usable
    /// graph, and one with a single node has no edges to make at all.
    #[test]
    fn test_tiny_partitions() {
        if !amx_ready() {
            return;
        }
        let params = test_params();
        for n in [1usize, 2, 5] {
            let storage = f16_storage(n, 64, 17, DistanceType::Dot);
            let hnsw = build_from_exact_knn(&storage, &params).unwrap().unwrap();
            assert_eq!(hnsw.len(), n);
            let nodes = hnsw.nodes().unwrap();
            assert_eq!(nodes[0].level_neighbors[0].len(), n - 1);
            let view = ImmutableHnswBottomView::new(&nodes);
            assert_eq!(Graph::len(&view), n);
        }
    }
}
