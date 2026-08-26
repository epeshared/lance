// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Exact top-k neighbours for every vector of one partition, computed on the
//! AMX-FP16 GEMM.
//!
//! HNSW builds its graph one node at a time, and every insertion pays a beam
//! search to find that node's candidate neighbours. When the partition is small
//! enough to hold in memory, scoring the whole `n x n` distance matrix on the
//! tile kernel and reading each node's neighbours straight out of it is
//! cheaper. This module produces that lookup table; turning it into a graph is
//! the caller's job.
//!
//! ## Shape of the computation
//!
//! The full matrix is never materialised — at `n = 360_000` it is 518 GB. The
//! matrix is walked in `ROW_BLOCK x COL_BLOCK` tiles, each tile is folded into
//! the running top-k of the nodes it touches, and then dropped. Only the top-k
//! state (`n * k * 8` bytes) and the packed vectors survive across tiles.
//!
//! Dot products are symmetric, so only the tiles at or above the diagonal are
//! computed; each one updates the top-k of *both* its rows and its columns.
//! That halves the GEMM work, at the cost of keeping every node's top-k live at
//! once (which the streaming requirement above already forces).
//!
//! ## Distance convention
//!
//! The kernel returns raw dot products, where larger is nearer. The top-k is
//! accumulated in that form and converted on the way out to the `Dot` distance
//! the rest of the crate uses — `1.0 - dot`, matching `dot_distance` in
//! `lance_linalg` and the `Dot` branch of the flat storage's distance
//! calculator — so smaller is nearer in the returned distances and the
//! neighbour lists are sorted ascending.
//!
//! ## Threading
//!
//! One call is single-threaded on purpose: the symmetric walk needs unshared
//! mutable access to every node's top-k, and the intended deployment runs one
//! partition per thread rather than many threads per partition.

use half::f16;
use lance_core::{Error, Result};
use lance_linalg::distance::DistanceType;
use lance_linalg::distance::dot_f16::PackedCentroidsF16;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _CMP_GT_OQ, _mm512_mask_cmp_ps_mask, _mm512_maskz_loadu_ps, _mm512_set1_ps,
};

/// Vectors scored against one packed column block per GEMM call.
///
/// Measured on a 128-thread, one-partition-per-thread run at `n = 10_000`,
/// `dim = 768`: 256 columns sustains 20.4 G pairs/s, 1024 drops to 11.6 G and
/// 8192 to 8.3 G — the block's scores have to stay resident in the private
/// cache while the next row block streams past them. The effect is only ~10%
/// single-threaded, so it cannot be tuned on one core.
const COL_BLOCK: usize = 256;

/// Rows handed to one GEMM call.
///
/// Each call reloads the tile configuration and releases it again, so the block
/// also has to be large enough for that to disappear. It does with room to
/// spare: the pair costs 133 cycles measured on Granite Rapids, against ~2M
/// cycles of tile arithmetic in a `1024 x 256 x 768` call. Shrinking the block
/// to 32 rows — 32x the calls — costs 7% at `n = 50_000`, and that is loop and
/// cache overhead rather than the reconfiguration. 256 and 1024 measure within
/// noise of each other.
const ROW_BLOCK: usize = 1024;

/// Exact nearest neighbours for every node of a partition.
///
/// `ids` and `distances` are both `num_nodes * k` long, laid out node-major:
/// node `i` owns `[i * k, (i + 1) * k)` in each. Within a node the neighbours
/// are sorted nearest-first (ascending distance), and no node lists itself.
#[derive(Debug, Clone, PartialEq)]
pub struct ExactKnn {
    /// Nodes covered — the `n` passed to [`exact_knn_topk`].
    num_nodes: usize,
    /// Neighbours stored per node. This is `min(k, n - 1)`, not the requested
    /// `k`: a node has only `n - 1` other nodes to choose from, so a partition
    /// smaller than `k + 1` yields shorter lists (and `0` for `n <= 1`).
    k: usize,
    ids: Vec<u32>,
    distances: Vec<f32>,
}

impl ExactKnn {
    /// Node `node`'s neighbour ids and their `Dot` distances (`1.0 - dot`),
    /// nearest-first. Both slices are [`Self::k`] long.
    pub fn neighbours(&self, node: usize) -> (&[u32], &[f32]) {
        let range = node * self.k..(node + 1) * self.k;
        (&self.ids[range.clone()], &self.distances[range])
    }
}

/// Computes the exact top-`k` nearest neighbours of every vector in `vectors`,
/// a row-major `[n, dim]` partition.
///
/// Only `f16` vectors under [`DistanceType::Dot`] are implemented, and only on a
/// host and build where the AMX-FP16 GEMM is usable; every other combination
/// returns [`Error::NotSupported`] so the caller can fall back to incremental
/// insertion. See the module docs for the blocking scheme and memory profile.
///
/// Runs on the calling thread. Peak allocation is `n * k * 8` bytes for the
/// top-k state plus roughly `2 * n * dim * 2` bytes for the packed vectors.
pub fn exact_knn_topk(
    vectors: &[f16],
    n: usize,
    dim: usize,
    k: usize,
    distance_type: DistanceType,
) -> Result<ExactKnn> {
    exact_knn_topk_blocked(vectors, n, dim, k, distance_type, true)
}

/// [`exact_knn_topk`], with the symmetry optimisation switchable.
///
/// `use_symmetry == false` computes every block of the matrix and folds it into
/// its rows only. It is strictly more work for the same answer, and exists so
/// the tests can hold the halved walk — where a block updates rows *and*
/// columns and the diagonal block must not be counted twice — against a version
/// with no such bookkeeping.
pub fn exact_knn_topk_blocked(
    vectors: &[f16],
    n: usize,
    dim: usize,
    k: usize,
    distance_type: DistanceType,
    use_symmetry: bool,
) -> Result<ExactKnn> {
    if distance_type != DistanceType::Dot {
        return Err(Error::not_supported(format!(
            "exact_knn_topk: only DistanceType::Dot is implemented, got {distance_type}"
        )));
    }
    let expected = n.checked_mul(dim).ok_or_else(|| {
        Error::invalid_input(format!(
            "exact_knn_topk: n ({n}) * dim ({dim}) overflows usize"
        ))
    })?;
    if vectors.len() != expected {
        return Err(Error::invalid_input(format!(
            "exact_knn_topk: vectors holds {} values, expected n ({n}) * dim ({dim}) = {expected}",
            vectors.len()
        )));
    }
    if n > u32::MAX as usize {
        return Err(Error::invalid_input(format!(
            "exact_knn_topk: n ({n}) exceeds u32::MAX, so neighbour ids would not fit"
        )));
    }

    // A node can have at most `n - 1` neighbours other than itself, so a
    // partition smaller than `k + 1` gets shorter lists rather than an error;
    // `ExactKnn::k` reports what was actually produced.
    let k = k.min(n.saturating_sub(1));
    if k == 0 {
        return Ok(ExactKnn {
            num_nodes: n,
            k: 0,
            ids: Vec::new(),
            distances: Vec::new(),
        });
    }

    // Every column block is packed once here and reused by every row block; at
    // 8-16% of a partition's time, packing is far too expensive to redo in the
    // inner loop.
    let mut blocks = Vec::with_capacity(n.div_ceil(COL_BLOCK));
    let mut start = 0;
    while start < n {
        let len = COL_BLOCK.min(n - start);
        let packed = PackedCentroidsF16::new(&vectors[start * dim..(start + len) * dim], len, dim)
            .ok_or_else(|| {
                Error::not_supported(format!(
                    "exact_knn_topk: the AMX-FP16 GEMM is unavailable for n = {n}, dim = {dim} \
                     (no kernel in this build, host without AMX-FP16, or LANCE_DISABLE_AMX set)"
                ))
            })?;
        blocks.push(ColumnBlock {
            start,
            len,
            stride: packed.num_centroids_padded(),
            packed,
        });
        start += len;
    }

    let mut heaps = TopKHeaps::new(n, k)?;
    let mut scores = vec![0f32; ROW_BLOCK * COL_BLOCK];
    // One mask bit per score; the column direction needs one mask word per row
    // of a block, the row direction one per 16 columns, so the larger wins.
    let mut masks = vec![0u16; ROW_BLOCK.max(COL_BLOCK.div_ceil(16))];
    // Zero-padded copy of the final row block, materialised only when `n` is not
    // a multiple of 32: the kernel has no partial-tile path, so it must be given
    // a whole number of 32-row tiles to read.
    let mut padded_tail = Vec::new();

    let mut row_start = 0;
    while row_start < n {
        let rows = ROW_BLOCK.min(n - row_start);
        let tiled_rows = rows.next_multiple_of(32);
        let is_tail_padded = row_start + tiled_rows > n;
        if is_tail_padded {
            padded_tail.clear();
            padded_tail.resize(tiled_rows * dim, f16::ZERO);
            padded_tail[..rows * dim].copy_from_slice(&vectors[row_start * dim..]);
        }
        let data: &[f16] = if is_tail_padded {
            &padded_tail
        } else {
            &vectors[row_start * dim..]
        };

        for block in &blocks {
            // Below the diagonal: every pair in this block was already scored,
            // and folded into both of its endpoints, by the transposed block.
            if use_symmetry && block.start + block.len <= row_start {
                continue;
            }
            block
                .packed
                .score(data, tiled_rows, dim, &mut scores, block.stride);
            // Rows `rows..tiled_rows` and columns `block.len..block.stride` are
            // the kernel's zero padding. Neither is inside the ranges below, so
            // no padding slot can ever reach a neighbour list.
            let scored = ScoreBlock {
                scores: &scores[..rows * block.stride],
                stride: block.stride,
                row_start,
                rows,
                col_start: block.start,
                cols: block.len,
            };
            if use_symmetry {
                scan_upper_triangle(&scored, &mut heaps, &mut masks);
            } else {
                scan_rows(&scored, &mut heaps, &mut masks);
            }
        }
        row_start += ROW_BLOCK;
    }

    heaps.into_knn(n)
}

/// A contiguous run of the partition's vectors, packed once into the layout the
/// GEMM reads its B operand in.
struct ColumnBlock {
    /// Index of this block's first vector within the partition.
    start: usize,
    /// Vectors in the block, before the kernel's padding.
    len: usize,
    /// Scores the GEMM writes per row: `len` rounded up to a multiple of 32.
    stride: usize,
    packed: PackedCentroidsF16,
}

/// One scored sub-rectangle of the distance matrix: rows
/// `[row_start, row_start + rows)` against columns
/// `[col_start, col_start + cols)`, row-major with `stride` floats per row.
struct ScoreBlock<'a> {
    scores: &'a [f32],
    stride: usize,
    row_start: usize,
    rows: usize,
    col_start: usize,
    cols: usize,
}

/// A neighbour while the top-k is still being accumulated: the raw dot product
/// (larger is nearer, so the heap below is a min-heap on it) and its node.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    dot: f32,
    id: u32,
}

/// Every node's bounded top-k, live simultaneously.
///
/// Each node owns a `k`-slot min-heap on the dot product, so the root is the
/// weakest neighbour kept and a candidate is admitted iff it beats the root.
/// The roots are mirrored into a separate contiguous `thresholds` array because
/// the column-direction scan compares 16 *different* nodes' thresholds at once
/// and needs them in one vector load.
struct TopKHeaps {
    k: usize,
    /// `n * k` slots; node `i`'s heap is `[i * k, i * k + lens[i])`.
    slots: Vec<Candidate>,
    lens: Vec<u32>,
    /// Admission threshold per node: the heap root once full, `-inf` while
    /// filling (so every finite candidate is taken).
    thresholds: Vec<f32>,
}

impl TopKHeaps {
    fn new(n: usize, k: usize) -> Result<Self> {
        let slots = n.checked_mul(k).ok_or_else(|| {
            Error::invalid_input(format!(
                "exact_knn_topk: n ({n}) * k ({k}) neighbour slots overflows usize"
            ))
        })?;
        Ok(Self {
            k,
            slots: vec![
                Candidate {
                    dot: f32::NEG_INFINITY,
                    id: u32::MAX,
                };
                slots
            ],
            lens: vec![0; n],
            thresholds: vec![f32::NEG_INFINITY; n],
        })
    }

    /// Offers `id` at dot product `dot` to node `node`'s top-k, keeping it only
    /// if it beats the weakest neighbour already there.
    ///
    /// The threshold is rechecked here rather than trusted from the caller: the
    /// scans compare a whole row or column strip against one snapshot of it, so
    /// by the time a candidate is offered the bar may already have risen. The
    /// test is written the positive way round so a NaN dot product — which
    /// compares greater than nothing — is dropped rather than admitted.
    #[inline]
    fn offer(&mut self, node: usize, dot: f32, id: u32) {
        if dot > self.thresholds[node] {
            let base = node * self.k;
            let len = self.lens[node] as usize;
            if len < self.k {
                self.slots[base + len] = Candidate { dot, id };
                self.lens[node] = len as u32 + 1;
                sift_up(&mut self.slots[base..base + len + 1]);
                if len + 1 == self.k {
                    self.thresholds[node] = self.slots[base].dot;
                }
            } else {
                self.slots[base] = Candidate { dot, id };
                sift_down(&mut self.slots[base..base + self.k]);
                self.thresholds[node] = self.slots[base].dot;
            }
        }
    }

    /// Sorts each node's heap nearest-first and converts the dot products to
    /// `Dot` distances.
    fn into_knn(mut self, n: usize) -> Result<ExactKnn> {
        let k = self.k;
        let mut ids = vec![0u32; n * k];
        let mut distances = vec![0f32; n * k];
        for node in 0..n {
            let len = self.lens[node] as usize;
            if len != k {
                // Every node is offered all `n - 1` others and `k <= n - 1`, so
                // a short heap means candidates were rejected for not comparing
                // greater than `-inf` — i.e. the input contains NaN.
                return Err(Error::invalid_input(format!(
                    "exact_knn_topk: node {node} found only {len} of {k} neighbours; \
                     the partition's vectors must all be finite"
                )));
            }
            let heap = &mut self.slots[node * k..node * k + k];
            // Nearest-first, with the id breaking ties so the output does not
            // depend on the order candidates happened to arrive in.
            heap.sort_unstable_by(|a, b| b.dot.total_cmp(&a.dot).then(a.id.cmp(&b.id)));
            for (rank, candidate) in heap.iter().enumerate() {
                ids[node * k + rank] = candidate.id;
                // `Dot` distance is `1.0 - dot`; see the module docs.
                distances[node * k + rank] = 1.0 - candidate.dot;
            }
        }
        Ok(ExactKnn {
            num_nodes: n,
            k,
            ids,
            distances,
        })
    }
}

/// Restores the min-heap property after `heap`'s last slot was appended.
fn sift_up(heap: &mut [Candidate]) {
    let mut child = heap.len() - 1;
    while child > 0 {
        let parent = (child - 1) / 2;
        if heap[child].dot >= heap[parent].dot {
            break;
        }
        heap.swap(child, parent);
        child = parent;
    }
}

/// Restores the min-heap property after `heap`'s root was replaced.
fn sift_down(heap: &mut [Candidate]) {
    let mut parent = 0;
    loop {
        let left = 2 * parent + 1;
        if left >= heap.len() {
            break;
        }
        let right = left + 1;
        let smaller = if right < heap.len() && heap[right].dot < heap[left].dot {
            right
        } else {
            left
        };
        if heap[smaller].dot >= heap[parent].dot {
            break;
        }
        heap.swap(parent, smaller);
        parent = smaller;
    }
}

/// Folds a block into the top-k of its rows *and* its columns, counting each
/// unordered pair `(i, j)` exactly once by keeping only `i < j`.
///
/// The same filter serves both kinds of block the symmetric walk produces. In a
/// block strictly above the diagonal every `i` is already below every `j`, so it
/// is free; in a block straddling the diagonal it is what drops the self pairs
/// and the mirror images that the block's own upper half already covered.
/// How many rows a column strip scans before re-reading its per-column thresholds.
///
/// Only affects speed, never the result: a threshold rises monotonically as offers
/// land, so scanning against a stale (lower) bar merely lets more candidates through
/// the coarse filter to be rejected by `offer` itself. Smaller values keep the bar
/// fresher at the cost of shorter vectorised runs.
///
/// Calibrated at n=9499 and n=26656, 128 threads, k=150: 8 / 32 / 128 land within
/// 3-5% of each other, with 8 marginally ahead. The knob is not sensitive.
const THRESHOLD_REFRESH: usize = 8;

fn scan_upper_triangle(block: &ScoreBlock, heaps: &mut TopKHeaps, masks: &mut [u16]) {
    // Row direction: node `i` against the columns `j > i`.
    for row in 0..block.rows {
        let i = block.row_start + row;
        let first = (i + 1).saturating_sub(block.col_start);
        if first >= block.cols {
            // `i` only grows from here, so no later row has eligible columns.
            break;
        }
        let offset = row * block.stride;
        let dots = &block.scores[offset + first..offset + block.cols];
        let words = dots.len().div_ceil(16);
        beats_threshold(dots, heaps.thresholds[i], &mut masks[..words]);
        for (word, &bits) in masks[..words].iter().enumerate() {
            let mut bits = bits;
            while bits != 0 {
                let lane = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let col = word * 16 + lane;
                heaps.offer(i, dots[col], (block.col_start + first + col) as u32);
            }
        }
    }

    // Column direction: node `j` against the rows `i < j`. Sixteen columns are
    // carried at a time so the loads stay contiguous — the transpose lives in
    // the thresholds vector, not in the memory access pattern.
    let mut strip = 0;
    while strip < block.cols {
        let width = 16.min(block.cols - strip);
        let last_col = block.col_start + strip + width;
        // Rows at or past the strip's last column have no `i < j` lane left.
        let rows = block.rows.min(last_col.saturating_sub(block.row_start));
        if rows == 0 {
            break;
        }
        // Re-read the strip's thresholds every `THRESHOLD_REFRESH` rows rather
        // than once for the whole strip. A column's bar rises as the rows go by,
        // and every row compared against a stale bar turns into an `offer` that
        // is rejected after the fact — cheap individually, but the dominant cost
        // once `k` is large enough for the heap traffic to outweigh the GEMM.
        let mut first_row = 0;
        while first_row < rows {
            let batch = THRESHOLD_REFRESH.min(rows - first_row);
            let thresholds = &heaps.thresholds[block.col_start + strip..last_col];
            let scanned =
                &block.scores[first_row * block.stride..(first_row + batch) * block.stride];
            beats_lane_thresholds(
                scanned,
                block.stride,
                strip,
                thresholds,
                &mut masks[..batch],
            );
            for (offset, &bits) in masks[..batch].iter().enumerate() {
                let row = first_row + offset;
                let i = block.row_start + row;
                let mut bits = bits & lanes_above(i, block.col_start + strip, width);
                while bits != 0 {
                    let lane = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let j = block.col_start + strip + lane;
                    heaps.offer(j, block.scores[row * block.stride + strip + lane], i as u32);
                }
            }
            first_row += batch;
        }
        strip += 16;
    }
}

/// Folds a block into the top-k of its rows only, dropping the self pair. Used
/// by the non-symmetric reference walk, which computes every block.
fn scan_rows(block: &ScoreBlock, heaps: &mut TopKHeaps, masks: &mut [u16]) {
    for row in 0..block.rows {
        let i = block.row_start + row;
        let offset = row * block.stride;
        let dots = &block.scores[offset..offset + block.cols];
        let words = dots.len().div_ceil(16);
        beats_threshold(dots, heaps.thresholds[i], &mut masks[..words]);
        // Under Dot a node's own score is its largest, so without this every
        // node's nearest neighbour would be itself.
        if i >= block.col_start && i < block.col_start + block.cols {
            let self_col = i - block.col_start;
            masks[self_col / 16] &= !(1 << (self_col % 16));
        }
        for (word, &bits) in masks[..words].iter().enumerate() {
            let mut bits = bits;
            while bits != 0 {
                let lane = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let col = word * 16 + lane;
                heaps.offer(i, dots[col], (block.col_start + col) as u32);
            }
        }
    }
}

/// Bit mask selecting the low `len` lanes of a 16-wide vector.
#[inline]
fn low_lanes(len: usize) -> u16 {
    if len >= 16 {
        u16::MAX
    } else {
        ((1u32 << len) - 1) as u16
    }
}

/// Lanes of a `width`-wide column strip starting at node `first_col` whose
/// column index is strictly greater than `row`.
#[inline]
fn lanes_above(row: usize, first_col: usize, width: usize) -> u16 {
    let skip = (row + 1).saturating_sub(first_col);
    if skip >= width {
        0
    } else {
        low_lanes(width) & !low_lanes(skip)
    }
}

/// Bit `l` of `masks[w]` is set iff `dots[w * 16 + l] > threshold`; bits past
/// the end of `dots` are cleared. `masks` must be `dots.len().div_ceil(16)`
/// long.
///
/// This comparison is the whole cost of the top-k scan in steady state — a
/// candidate that beats the current k-th neighbour is rare, so almost every
/// score is touched exactly once, here. The crate is compiled for the x86-64
/// baseline, so leaving it to the optimiser yields scalar code and costs 25-30%
/// of end-to-end throughput; the explicit AVX-512 path brings that to 6%.
#[inline]
fn beats_threshold(dots: &[f32], threshold: f32, masks: &mut [u16]) {
    debug_assert_eq!(masks.len(), dots.len().div_ceil(16));
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f") {
        // SAFETY: avx512f was just detected, which is the only requirement.
        unsafe { beats_threshold_avx512(dots, threshold, masks) };
        return;
    }
    beats_threshold_baseline(dots, threshold, masks)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn beats_threshold_avx512(dots: &[f32], threshold: f32, masks: &mut [u16]) {
    let broadcast = _mm512_set1_ps(threshold);
    for (chunk, mask) in dots.chunks(16).zip(masks.iter_mut()) {
        let valid = low_lanes(chunk.len());
        // SAFETY: the masked load reads only the `chunk.len()` lanes `valid`
        // selects, all of which are inside `chunk`.
        let scores = unsafe { _mm512_maskz_loadu_ps(valid, chunk.as_ptr()) };
        *mask = _mm512_mask_cmp_ps_mask::<_CMP_GT_OQ>(valid, scores, broadcast);
    }
}

fn beats_threshold_baseline(dots: &[f32], threshold: f32, masks: &mut [u16]) {
    for (chunk, mask) in dots.chunks(16).zip(masks.iter_mut()) {
        let mut bits = 0u16;
        for (lane, &dot) in chunk.iter().enumerate() {
            if dot > threshold {
                bits |= 1 << lane;
            }
        }
        *mask = bits;
    }
}

/// Bit `l` of `masks[r]` is set iff row `r` of `block` beats `thresholds[l]` in
/// column `strip + l`. `block` must hold exactly `masks.len()` rows of `stride`
/// floats, and `thresholds` at most 16 entries ending within a row.
///
/// The column-direction twin of [`beats_threshold`]: one comparison per score,
/// against a *different* threshold per lane, with the loads still contiguous.
#[inline]
fn beats_lane_thresholds(
    block: &[f32],
    stride: usize,
    strip: usize,
    thresholds: &[f32],
    masks: &mut [u16],
) {
    debug_assert_eq!(block.len(), masks.len() * stride);
    debug_assert!(strip + thresholds.len() <= stride);
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx512f") {
        // SAFETY: avx512f was just detected, which is the only requirement.
        unsafe { beats_lane_thresholds_avx512(block, stride, strip, thresholds, masks) };
        return;
    }
    beats_lane_thresholds_baseline(block, stride, strip, thresholds, masks)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn beats_lane_thresholds_avx512(
    block: &[f32],
    stride: usize,
    strip: usize,
    thresholds: &[f32],
    masks: &mut [u16],
) {
    let valid = low_lanes(thresholds.len());
    // SAFETY: the masked loads read only the `thresholds.len()` lanes `valid`
    // selects; `thresholds` is that long and every row of `block` has at least
    // `strip + thresholds.len()` floats (checked by the caller's debug asserts).
    unsafe {
        let bars = _mm512_maskz_loadu_ps(valid, thresholds.as_ptr());
        for (row, mask) in block.chunks_exact(stride).zip(masks.iter_mut()) {
            let scores = _mm512_maskz_loadu_ps(valid, row[strip..].as_ptr());
            *mask = _mm512_mask_cmp_ps_mask::<_CMP_GT_OQ>(valid, scores, bars);
        }
    }
}

fn beats_lane_thresholds_baseline(
    block: &[f32],
    stride: usize,
    strip: usize,
    thresholds: &[f32],
    masks: &mut [u16],
) {
    for (row, mask) in block.chunks_exact(stride).zip(masks.iter_mut()) {
        let mut bits = 0u16;
        for (lane, &bar) in thresholds.iter().enumerate() {
            if row[strip + lane] > bar {
                bits |= 1 << lane;
            }
        }
        *mask = bits;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::{FixedSizeListArray, Float16Array};
    use arrow_schema::{DataType, Field};
    use rayon::prelude::*;
    use std::sync::Arc;

    use crate::vector::flat::storage::FlatFloatStorage;
    use crate::vector::storage::{DistCalculator, VectorStore};

    /// The tile kernel is compiled in only where `build.rs` found a compiler for
    /// it, and runs only on a host that grants XTILEDATA, so every test that
    /// needs a real result returns early elsewhere. Skipping is the same
    /// convention the flat storage's AMX test uses.
    /// Whether `exact_knn_topk` will find a kernel, which is what
    /// `PackedCentroidsF16::new` asks — hardware support alone.
    ///
    /// Deliberately *not* `amx_fp16_available()`: that adds the
    /// `LANCE_DISABLE_AMX` kill switch, which this path does not consult, so
    /// under `LANCE_DISABLE_AMX=1` the tests below would demand a `NotSupported`
    /// error from a call that still succeeds.
    ///
    /// TODO: decide whether that is the intended behaviour. The switch is
    /// documented as taking the AMX paths out of service, and partition
    /// assignment honours it; exact-KNN graph building not honouring it means an
    /// operator cannot A/B this path or fall back without a rebuild. Changing it
    /// is a behaviour change and belongs in its own PR.
    fn amx_ready() -> bool {
        lance_linalg::distance::dot_f16::amx_fp16_supported()
    }

    /// `n` deterministic pseudo-random unit vectors, row-major.
    ///
    /// Unit-normalised in f32 before rounding to f16, so dot products land in a
    /// range f16 represents well and exact ties between distinct pairs are
    /// vanishingly unlikely — which is what lets the tests compare neighbour
    /// *ids* and not just distances.
    fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<f16> {
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
            out.extend(v.into_iter().map(f16::from_f32));
        }
        out
    }

    /// `Dot` distance of two of the partition's vectors, straight from the
    /// definition in f32.
    fn pair_distance(vectors: &[f16], dim: usize, i: usize, j: usize) -> f32 {
        let x = &vectors[i * dim..(i + 1) * dim];
        let y = &vectors[j * dim..(j + 1) * dim];
        1.0 - x
            .iter()
            .zip(y)
            .map(|(a, b)| a.to_f32() * b.to_f32())
            .sum::<f32>()
    }

    /// Ground truth by the definition: every pair scored in f32, self excluded,
    /// sorted nearest-first with ids breaking ties.
    fn brute_force(vectors: &[f16], n: usize, dim: usize, k: usize) -> (Vec<u32>, Vec<f32>) {
        let rows: Vec<(Vec<u32>, Vec<f32>)> = (0..n)
            .into_par_iter()
            .map(|i| {
                let mut scored: Vec<(f32, u32)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| (pair_distance(vectors, dim, i, j), j as u32))
                    .collect();
                scored.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                scored.truncate(k);
                (
                    scored.iter().map(|s| s.1).collect(),
                    scored.iter().map(|s| s.0).collect(),
                )
            })
            .collect();
        let mut ids = Vec::with_capacity(n * k);
        let mut distances = Vec::with_capacity(n * k);
        for (row_ids, row_distances) in rows {
            ids.extend(row_ids);
            distances.extend(row_distances);
        }
        (ids, distances)
    }

    /// Holds a result against a plain pairwise f32 loop.
    ///
    /// Three properties, all at once: the reported distance really is the
    /// reported neighbour's, that neighbour really is as near as the true
    /// rank-`r` neighbour, and no neighbour repeats. Together they say the list
    /// *is* an exact top-k, which is the guarantee — the ids alone cannot be
    /// demanded because the tile kernel accumulates in a different order than a
    /// sequential loop, so two candidates whose distances differ by an f32 ulp
    /// can legitimately swap ranks.
    ///
    /// Such swaps are rare enough that the test still insists on near-total id
    /// agreement; without that floor the tolerance would excuse a genuinely
    /// reshuffled result.
    fn assert_is_exact_knn(knn: &ExactKnn, vectors: &[f16], n: usize, dim: usize, k: usize) {
        const TOLERANCE: f32 = 1e-5;
        assert_eq!(knn.num_nodes, n);
        assert_eq!(knn.k, k);
        let (want_ids, want_distances) = brute_force(vectors, n, dim, k);
        let mut exact_ids = 0usize;
        for node in 0..n {
            let (ids, distances) = knn.neighbours(node);
            let want_id = &want_ids[node * k..(node + 1) * k];
            let want_distance = &want_distances[node * k..(node + 1) * k];
            let mut seen = ids.to_vec();
            seen.sort_unstable();
            seen.dedup();
            assert_eq!(seen.len(), k, "node {node} repeats a neighbour: {ids:?}");
            for rank in 0..k {
                let truth = pair_distance(vectors, dim, node, ids[rank] as usize);
                assert!(
                    (distances[rank] - truth).abs() <= TOLERANCE,
                    "node {node} rank {rank}: reported {} for neighbour {}, which is at {truth}",
                    distances[rank],
                    ids[rank]
                );
                assert!(
                    (truth - want_distance[rank]).abs() <= TOLERANCE,
                    "node {node} rank {rank}: neighbour {} is at {truth}, but the true rank-{rank} \
                     neighbour {} is at {}",
                    ids[rank],
                    want_id[rank],
                    want_distance[rank]
                );
                exact_ids += usize::from(ids[rank] == want_id[rank]);
            }
        }
        assert!(
            exact_ids * 1000 >= n * k * 999,
            "only {exact_ids} of {} neighbour slots matched the reference id exactly; \
             ulp-level ties cannot account for that many",
            n * k
        );
    }

    /// The headline check, at the shape the partitions actually have.
    #[test]
    fn test_matches_brute_force() {
        if !amx_ready() {
            return;
        }
        let (n, dim, k) = (1000, 768, 32);
        let vectors = random_vectors(n, dim, 7);
        let knn = exact_knn_topk(&vectors, n, dim, k, DistanceType::Dot).unwrap();
        assert_is_exact_knn(&knn, &vectors, n, dim, k);
    }

    /// The `dim % 32` tail is a separate scalar path inside the kernel and is
    /// not covered by the packed layout, so it gets its own comparison. `dim`
    /// below 32 leaves *only* that path.
    #[test]
    fn test_matches_brute_force_with_unaligned_dim() {
        if !amx_ready() {
            return;
        }
        for (n, dim, k) in [(300usize, 100usize, 8usize), (300, 20, 8)] {
            let vectors = random_vectors(n, dim, 11);
            let knn = exact_knn_topk(&vectors, n, dim, k, DistanceType::Dot).unwrap();
            assert_is_exact_knn(&knn, &vectors, n, dim, k);
        }
    }

    /// More than one row block, and a partial one at the end: the row-blocking
    /// arithmetic and the zero-padded tail block only appear above `ROW_BLOCK`.
    #[test]
    fn test_matches_brute_force_across_row_blocks() {
        if !amx_ready() {
            return;
        }
        let (n, dim, k) = (2600, 64, 12);
        let vectors = random_vectors(n, dim, 13);
        let knn = exact_knn_topk(&vectors, n, dim, k, DistanceType::Dot).unwrap();
        assert_is_exact_knn(&knn, &vectors, n, dim, k);
    }

    /// The kernel is fed 32-row and 32-column tiles, so any `n` that is not a
    /// multiple of 32 is scored against zero padding. A padding slot scores 0.0,
    /// which beats every genuinely negative dot product, so it must never reach
    /// a neighbour list — it would be an edge to a node that does not exist.
    #[test]
    fn test_no_padding_ids_when_n_is_unaligned() {
        if !amx_ready() {
            return;
        }
        let (n, dim, k) = (1001, 64, 16);
        let vectors = random_vectors(n, dim, 3);
        let knn = exact_knn_topk(&vectors, n, dim, k, DistanceType::Dot).unwrap();
        assert_eq!(knn.ids.len(), n * k);
        for (slot, &id) in knn.ids.iter().enumerate() {
            assert!(
                (id as usize) < n,
                "slot {slot} points at node {id}, outside the partition's {n} nodes"
            );
        }
    }

    /// A node's own score is its largest under Dot, so a missing self-exclusion
    /// would make every node its own nearest neighbour.
    #[test]
    fn test_never_returns_self() {
        if !amx_ready() {
            return;
        }
        let (n, dim, k) = (600, 64, 12);
        let vectors = random_vectors(n, dim, 5);
        let knn = exact_knn_topk(&vectors, n, dim, k, DistanceType::Dot).unwrap();
        for node in 0..n {
            let (ids, _) = knn.neighbours(node);
            assert!(
                !ids.contains(&(node as u32)),
                "node {node} lists itself: {ids:?}"
            );
        }
    }

    /// The symmetric walk skips half the matrix and folds each block into both
    /// its rows and its columns; the diagonal block must contribute once, not
    /// twice. Held against the walk that computes everything and only ever
    /// updates rows.
    #[test]
    fn test_symmetric_walk_matches_full_walk() {
        if !amx_ready() {
            return;
        }
        // Spans several row blocks and lands on a partial block in both
        // directions, so diagonal, above-diagonal and skipped blocks all occur.
        for &(n, dim, k) in &[(2500usize, 64usize, 10usize), (517, 96, 5), (33, 32, 7)] {
            let vectors = random_vectors(n, dim, 17);
            let symmetric =
                exact_knn_topk_blocked(&vectors, n, dim, k, DistanceType::Dot, true).unwrap();
            let full =
                exact_knn_topk_blocked(&vectors, n, dim, k, DistanceType::Dot, false).unwrap();
            assert_eq!(symmetric.ids, full.ids, "ids differ at n = {n}");
            assert_eq!(
                symmetric.distances, full.distances,
                "distances differ at n = {n}"
            );
        }
    }

    /// Pins the returned distance to what the rest of the crate means by `Dot`,
    /// rather than to this module's own idea of it: the flat storage's distance
    /// calculator is the consumer HNSW already scores neighbours with.
    #[test]
    fn test_distance_matches_flat_storage() {
        if !amx_ready() {
            return;
        }
        let (n, dim, k) = (200, 64, 4);
        let vectors = random_vectors(n, dim, 23);
        let knn = exact_knn_topk(&vectors, n, dim, k, DistanceType::Dot).unwrap();

        let values = Float16Array::from_iter_values(vectors.iter().copied());
        let field = Arc::new(Field::new("item", DataType::Float16, true));
        let list = FixedSizeListArray::try_new(field, dim as i32, Arc::new(values), None).unwrap();
        let storage = FlatFloatStorage::new(list, DistanceType::Dot);

        for node in 0..n {
            let calculator = storage.dist_calculator_from_id(node as u32);
            let (ids, distances) = knn.neighbours(node);
            let mut previous = f32::NEG_INFINITY;
            for (&id, &distance) in ids.iter().zip(distances) {
                let want = calculator.distance(id);
                assert!(
                    (distance - want).abs() <= 1e-3,
                    "node {node} -> {id}: got {distance}, storage says {want}"
                );
                // Nearest-first means ascending under this convention.
                assert!(
                    distance >= previous,
                    "node {node} is not sorted nearest-first"
                );
                previous = distance;
            }
        }
    }

    /// A partition with fewer than `k + 1` vectors cannot fill `k` neighbours,
    /// so the lists shrink to what exists instead of failing or padding.
    #[test]
    fn test_k_larger_than_partition() {
        if !amx_ready() {
            return;
        }
        let (n, dim) = (5, 64);
        let vectors = random_vectors(n, dim, 29);
        let knn = exact_knn_topk(&vectors, n, dim, 150, DistanceType::Dot).unwrap();
        assert_eq!(knn.k, n - 1);
        assert_eq!(knn.ids.len(), n * (n - 1));
        for node in 0..n {
            let (ids, _) = knn.neighbours(node);
            let mut seen: Vec<u32> = ids.to_vec();
            seen.sort_unstable();
            let want: Vec<u32> = (0..n as u32).filter(|&j| j != node as u32).collect();
            assert_eq!(
                seen, want,
                "node {node} must see every other node exactly once"
            );
        }
    }

    #[test]
    fn test_degenerate_partitions() {
        for n in [0usize, 1] {
            let vectors = random_vectors(n, 64, 31);
            let knn = exact_knn_topk(&vectors, n, 64, 10, DistanceType::Dot).unwrap();
            assert_eq!(knn.num_nodes, n);
            assert_eq!(knn.k, 0);
            assert!(knn.ids.is_empty());
            assert!(knn.distances.is_empty());
        }
    }

    /// `k = 0` is degenerate rather than invalid, and must not reach the kernel.
    #[test]
    fn test_zero_k() {
        let vectors = random_vectors(64, 32, 37);
        let knn = exact_knn_topk(&vectors, 64, 32, 0, DistanceType::Dot).unwrap();
        assert_eq!(knn.k, 0);
        assert!(knn.ids.is_empty());
    }

    /// Without the kernel there is no partial mode: the caller has to be told
    /// to keep inserting incrementally, and told it as an error it can match on
    /// rather than a panic or a silently degraded answer.
    #[test]
    fn test_reports_unavailable_kernel() {
        let (n, dim) = (64, 64);
        let vectors = random_vectors(n, dim, 47);
        let result = exact_knn_topk(&vectors, n, dim, 8, DistanceType::Dot);
        if amx_ready() {
            assert!(result.is_ok(), "AMX is available but got {result:?}");
            return;
        }
        let error = result.unwrap_err();
        assert!(
            matches!(error, Error::NotSupported { .. }),
            "expected NotSupported without the AMX-FP16 kernel, got {error}"
        );
        assert!(
            error.to_string().contains("AMX-FP16 GEMM is unavailable"),
            "unexpected message: {error}"
        );
    }

    #[test]
    fn test_rejects_other_distance_types() {
        let vectors = random_vectors(64, 32, 41);
        for distance_type in [DistanceType::L2, DistanceType::Cosine] {
            let error = exact_knn_topk(&vectors, 64, 32, 4, distance_type).unwrap_err();
            assert!(
                matches!(error, Error::NotSupported { .. }),
                "expected NotSupported for {distance_type}, got {error}"
            );
            assert!(
                error.to_string().contains("only DistanceType::Dot"),
                "unexpected message: {error}"
            );
        }
    }

    #[test]
    fn test_rejects_mismatched_length() {
        let vectors = random_vectors(64, 32, 43);
        let error = exact_knn_topk(&vectors, 65, 32, 4, DistanceType::Dot).unwrap_err();
        assert!(
            matches!(error, Error::InvalidInput { .. }),
            "expected InvalidInput, got {error}"
        );
        assert!(
            error.to_string().contains("2048 values"),
            "message should name the actual length: {error}"
        );
    }

    /// The AVX-512 scan and the portable fallback must agree bit for bit; they
    /// are the same comparison and only one of them ever runs on a given host.
    #[test]
    fn test_mask_kernels_agree() {
        let dots: Vec<f32> = (0..37).map(|i| (i as f32) * 0.25 - 4.0).collect();
        for threshold in [f32::NEG_INFINITY, -4.0, 0.0, 3.0, 100.0] {
            let words = dots.len().div_ceil(16);
            let mut dispatched = vec![0u16; words];
            let mut baseline = vec![0u16; words];
            beats_threshold(&dots, threshold, &mut dispatched);
            beats_threshold_baseline(&dots, threshold, &mut baseline);
            assert_eq!(dispatched, baseline, "threshold {threshold}");
            // Bits past the end of `dots` must stay clear, or the scan would
            // read neighbours out of the block's padding.
            assert_eq!(
                dispatched[words - 1] & !low_lanes(dots.len() % 16),
                0,
                "tail bits set for threshold {threshold}"
            );
        }

        let (stride, rows, strip, width) = (48usize, 5usize, 32usize, 11usize);
        let block: Vec<f32> = (0..rows * stride).map(|i| (i % 17) as f32 - 8.0).collect();
        let thresholds: Vec<f32> = (0..width).map(|l| l as f32 - 5.0).collect();
        let mut dispatched = vec![0u16; rows];
        let mut baseline = vec![0u16; rows];
        beats_lane_thresholds(&block, stride, strip, &thresholds, &mut dispatched);
        beats_lane_thresholds_baseline(&block, stride, strip, &thresholds, &mut baseline);
        assert_eq!(dispatched, baseline);
        for (row, &bits) in dispatched.iter().enumerate() {
            assert_eq!(bits & !low_lanes(width), 0, "row {row} set a padding lane");
        }
    }

    #[test]
    fn test_lanes_above() {
        // Column strip [10, 16): row 12 may only see columns 13..16.
        assert_eq!(lanes_above(12, 10, 6), 0b111000);
        // Every column is above the row.
        assert_eq!(lanes_above(3, 10, 6), 0b111111);
        // No column is.
        assert_eq!(lanes_above(20, 10, 6), 0);
        // The diagonal itself is excluded, not included.
        assert_eq!(lanes_above(10, 10, 6), 0b111110);
    }
}
