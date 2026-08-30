// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow::{
    array::{ArrayData, make_array},
    buffer::Buffer,
    compute::cast,
};
use arrow_array::types::{Float16Type, Float32Type, Float64Type};
use arrow_array::{Array, ArrayRef, BooleanArray, FixedSizeListArray, cast::AsArray};
use arrow_schema::{DataType, Field};
use lance_arrow::{BufferExt, DataTypeExt, FixedSizeListArrayExt};
use lance_core::{Error, Result};
use lance_linalg::distance::DistanceType;
use lance_linalg::distance::dot_f16::{amx_fp16_available, amx_fp32_cast_available};
use log::{debug, warn};
use prost::bytes;
use std::sync::LazyLock;
use std::{ops::Range, sync::Arc};

use super::pb;
use crate::pb::Tensor;
use crate::vector::flat::storage::FlatBinStorage;
use crate::vector::flat::storage::FlatFloatStorage;
use crate::vector::hnsw::HNSW;
use crate::vector::hnsw::builder::{HnswBuildParams, HnswQueryParams};
use crate::vector::v3::subindex::IvfSubIndex;

enum SimpleIndexStatus {
    Auto,
    Enabled,
    Disabled,
}

static USE_HNSW_SPEEDUP_INDEXING: LazyLock<SimpleIndexStatus> = LazyLock::new(|| {
    if let Ok(v) = std::env::var("LANCE_USE_HNSW_SPEEDUP_INDEXING") {
        if v == "enabled" {
            SimpleIndexStatus::Enabled
        } else if v == "disabled" {
            SimpleIndexStatus::Disabled
        } else {
            SimpleIndexStatus::Auto
        }
    } else {
        SimpleIndexStatus::Auto
    }
});

/// Shape of the approximate assignment path: how wide the top-1 centroid lookup
/// searches, and how rich the graph it searches is.
///
/// Every vector in the dataset runs one `ef`-wide graph search to pick its
/// partition, so these are the knobs that trade assignment accuracy against
/// assignment cost on that path. They are read from the environment rather than
/// hardcoded because the case for exact assignment rests on a comparison against
/// them: "exact beats the graph lookup" is only interesting if the graph lookup
/// cannot buy the gap back more cheaply by searching wider, or by being built
/// better. Neither question can be asked while the values are literals.
///
/// The defaults are the historical ones -- `ef = 15` at query time over a graph
/// built at `ef_construction = 15` with 12 edges per node, which is a deliberately
/// minimal graph: it covers `num_centroids` nodes, not the dataset, so it is cheap
/// either way. Values that do not parse, or are zero, fall back to the default
/// rather than erroring; this is a measurement surface on a path that has no other
/// configuration, and failing an index build over a malformed environment variable
/// would be the worse outcome.
struct ApproxAssignParams {
    ef: usize,
    ef_construction: usize,
    num_edges: usize,
}

static APPROX_ASSIGN: LazyLock<ApproxAssignParams> = LazyLock::new(|| {
    fn read(var: &str, default: usize) -> usize {
        std::env::var(var)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(default)
    }
    let p = ApproxAssignParams {
        ef: read("LANCE_HNSW_ASSIGN_EF", 15),
        ef_construction: read("LANCE_HNSW_ASSIGN_EF_CONSTRUCTION", 15),
        num_edges: read("LANCE_HNSW_ASSIGN_EDGES", 12),
    };
    // Emitted unconditionally so an A/B run can confirm which arm it is in from
    // the log of every arm, including the one left on the defaults -- "no line"
    // is not evidence that the default was used. `info!` once the shape moves off
    // the defaults, because that changes index quality and a mislabeled arm is
    // worse than a log line; `debug!` otherwise. Once per process either way.
    let line = format!(
        "IVF partition assignment: approximate graph lookup ef={}, \
         ef_construction={}, num_edges={} (defaults 15/15/12)",
        p.ef, p.ef_construction, p.num_edges
    );
    if (p.ef, p.ef_construction, p.num_edges) == (15, 15, 12) {
        debug!("{line}");
    } else {
        log::info!("{line}");
    }
    p
});

/// How many centroid values (`num_centroids * dimension`) it takes before an
/// approximate index over the centroids pays for the cost of building it.
/// Benchmarked at 1024 centroids x 1024 dimensions, where it made assignment 2x
/// faster; below this the flat scan wins on its own.
const MIN_CENTROID_VALUES_FOR_INDEX: usize = 1_000_000;

/// Which routes onto the AMX-FP16 kernel are in service, as the two kill
/// switches leave them.
///
/// Passed in rather than read inside the routing functions so the routing table
/// can be tested by enumeration. Read from the environment, a test can only
/// assert that a function returns what the very function it calls returns —
/// which holds however the routing is wired, including wrongly.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct AmxAvailability {
    /// An f16 column may use the kernel directly: `amx_fp16_available()`.
    pub(crate) native: bool,
    /// An fp32 column may be rounded onto it: `amx_fp32_cast_available()`.
    /// Never true while `native` is false — `LANCE_DISABLE_AMX` outranks
    /// `LANCE_AMX_FP32_CAST` — but the routing code does not rely on that.
    pub(crate) rounded: bool,
}

impl AmxAvailability {
    /// What the two kill switches say in this process. The only place
    /// production routing reads them; everything below takes the answer as an
    /// argument.
    pub(crate) fn current() -> Self {
        Self {
            native: amx_fp16_available(),
            rounded: amx_fp32_cast_available(),
        }
    }
}

/// What stands between partition assignment and the exact flat path, or `None`
/// when that path is better served than an approximate lookup through this index
/// and nothing blocks it.
///
/// The reason is returned rather than folded into a `bool` because it is the
/// only thing that distinguishes the arms of an A/B run in the log. Whether the
/// fp32 cast was in play is expected *not* to show up in the k-means loss — that
/// is the result the cast exists to produce — so the loss cannot confirm which
/// path a build took, and the reason string is what does. It is also what an
/// operator asking "why did my fp32 build go approximate" needs; a `false` on
/// its own answers nothing.
///
/// The index turns one `M x N` problem -- every vector against every centroid --
/// into `M` independent top-1 graph searches. Each search walks its own path, so
/// no two vectors share a candidate set and the AMX-FP16 kernel behind them can
/// only ever score one query against a handful of neighbors: 32 MAC/cycle, one
/// of the tile's 16 output columns. Keeping the problem in its matrix shape lets
/// [`crate::vector::kmeans::compute_partitions`] reach the AMX-FP16 GEMM, which
/// fills all four accumulator tiles at 512 MAC/cycle.
///
/// Measured on 100M x 768 fp16 (dot, node-local, m=20, ef_construction=150), the
/// flat path won on both build time and recall at every `k` tried:
///
/// | k     | flat        | indexed     |
/// |-------|-------------|-------------|
/// | 10000 | 1965s / .980 | 2011s / .976 |
/// | 20000 | 2494s / .964 | 2588s / .832 |
/// | 40000 | 3730s / .970 | 3976s / .940 |
///
/// The recall gap is the larger effect: the graph lookup runs at `ef = 15`, so
/// some vectors land in a partition that is not their nearest and no `nprobes`
/// setting recovers them. Flat assignment is exact.
///
/// An fp32 column qualifies too, by rounding each block of vectors to f16 on the
/// way into the GEMM. Rounding moves some assignments, but far less than the
/// approximate graph lookup does: measured on 990k x 1536 real OpenAI embeddings
/// (dbpedia) against 2048 centroids, 0.231% of vectors landed on a different
/// centroid, the k-means loss moved by +0.0001%, and the vectors that moved paid
/// 7.9e-5 more distance against an average of 0.208 — four parts in ten
/// thousand. The data is nowhere near f16's range limits either (max |v| = 0.689
/// against a ceiling of 65504), with 0.27% of elements landing in the subnormal
/// range. Assignment then stays exact in a stronger sense than the graph lookup
/// can offer: the winner is picked from f16 scores, but the *distance* reported
/// for it is recomputed in fp32, so k-means' loss stays comparable with the
/// exact path's. Anything that does not round cleanly falls back per block.
///
/// The conditions below must stay in lockstep with the AMX gate in
/// `compute_membership_and_dist`; without the GEMM the flat path is ~2.7x slower
/// than the index (5361s vs 2011s at k=10000), so a mismatch here is expensive.
/// That includes both kill switches, which the two sides consult through the
/// same [`amx_fp16_available`] / [`amx_fp32_cast_available`] pair: an operator
/// turning either off has to move this decision too, or the build would take the
/// exact-assignment path with no GEMM under it.
///
/// The conditions are tested in the gate's own order and the first failure wins,
/// so an operator gets one answer rather than a list.
fn flat_amx_assignment_blocker(
    centroid_type: &DataType,
    num_centroids: usize,
    dimension: usize,
    distance_type: DistanceType,
    amx: AmxAvailability,
) -> Option<&'static str> {
    if distance_type != DistanceType::Dot {
        return Some("the distance type is not dot");
    }
    if dimension < 32 {
        return Some("the dimension is below one 32-wide k-pass");
    }
    if num_centroids < 32 {
        return Some("the centroid count is below one 32-centroid block");
    }
    match centroid_type {
        DataType::Float16 if amx.native => None,
        DataType::Float32 if amx.rounded => None,
        // Distinguishing the two switches matters: the fp32 one is the arm of
        // an A/B measurement, and reporting it as "AMX is off" would make the
        // control arm indistinguishable from a host that simply has no AMX.
        DataType::Float32 if amx.native => Some("LANCE_AMX_FP32_CAST is off"),
        DataType::Float16 | DataType::Float32 => Some(
            "AMX-FP16 is unavailable: no kernel in this build, no support on this CPU, or LANCE_DISABLE_AMX is on",
        ),
        _ => Some("the centroid type is neither float16 nor float32"),
    }
}

/// Which way one round of partition assignment goes, and why — the payload of
/// the `debug!` line in [`SimpleIndex::may_train_index`].
///
/// The three routes are not interchangeable: `flat+amx` is exact and fast,
/// `flat` is exact and ~2.7x slower, `hnsw` is approximate and loses recall no
/// `nprobes` setting recovers. Which one a build took is worth a line in the log
/// because nothing else reveals it — least of all the k-means loss, which the
/// fp32 cast is designed to leave unchanged.
struct AssignmentRoute {
    /// Whether to train the approximate centroid index.
    train_index: bool,
    /// `flat+amx`, `flat` or `hnsw`.
    route: &'static str,
    /// Why this route rather than another.
    reason: &'static str,
    /// What keeps the AMX-FP16 GEMM out from under assignment, or `None` when it
    /// is in play. Carried separately from `reason` because it is reported even
    /// when it did not decide the route: on a centroid set below
    /// [`MIN_CENTROID_VALUES_FOR_INDEX`] the route is flat either way, and this
    /// is then the only field that tells the arms of an A/B run apart.
    amx_blocker: Option<&'static str>,
}

/// Decides an [`AssignmentRoute`].
///
/// A separate function from [`SimpleIndex::may_train_index`] because what the
/// log *says* is worth asserting, and asserting it here needs no global logger
/// installed.
fn assignment_route(
    centroid_type: &DataType,
    num_centroids: usize,
    dimension: usize,
    distance_type: DistanceType,
    amx: AmxAvailability,
    status: &SimpleIndexStatus,
) -> AssignmentRoute {
    let amx_blocker =
        flat_amx_assignment_blocker(centroid_type, num_centroids, dimension, distance_type, amx);
    // Declining to build an index is not one outcome but two, and they perform
    // very differently, so the flat routes are named apart.
    let flat_route = if amx_blocker.is_none() {
        "flat+amx"
    } else {
        "flat"
    };
    let (train_index, route, reason) = match status {
        SimpleIndexStatus::Enabled => (true, "hnsw", "LANCE_USE_HNSW_SPEEDUP_INDEXING=enabled"),
        SimpleIndexStatus::Disabled => (
            false,
            flat_route,
            "LANCE_USE_HNSW_SPEEDUP_INDEXING=disabled",
        ),
        // Too few centroid values for the index to pay for itself, so assignment
        // stays flat whatever the GEMM can do for it. This is the branch every
        // small shape takes -- including 100 x 1536, where the gate below never
        // decides anything -- so without a reason of its own those builds would
        // log nothing at all.
        SimpleIndexStatus::Auto
            if num_centroids.saturating_mul(dimension) < MIN_CENTROID_VALUES_FOR_INDEX =>
        {
            (
                false,
                flat_route,
                "the centroid set is below the size an approximate index pays for",
            )
        }
        SimpleIndexStatus::Auto => match amx_blocker {
            Some(blocker) => (true, "hnsw", blocker),
            None => (
                false,
                flat_route,
                "the exact flat scan beats an approximate lookup on this shape",
            ),
        },
    };
    AssignmentRoute {
        train_index,
        route,
        reason,
        amx_blocker,
    }
}

#[derive(Debug)]
pub struct SimpleIndex {
    store: SimpleStore,
    index: HNSW,
}

#[derive(Debug)]
enum SimpleStore {
    Float(FlatFloatStorage),
    Binary(FlatBinStorage),
}

impl SimpleIndex {
    fn try_new(store: SimpleStore) -> Result<Self> {
        let hnsw = match &store {
            SimpleStore::Float(store) => HNSW::index_vectors(
                store,
                HnswBuildParams::default()
                    .ef_construction(APPROX_ASSIGN.ef_construction)
                    .num_edges(APPROX_ASSIGN.num_edges),
            )?,
            SimpleStore::Binary(store) => HNSW::index_vectors(
                store,
                HnswBuildParams::default()
                    .ef_construction(APPROX_ASSIGN.ef_construction)
                    .num_edges(APPROX_ASSIGN.num_edges),
            )?,
        };
        Ok(Self { store, index: hnsw })
    }

    // train HNSW over the centroids to speed up finding the nearest clusters,
    // only train if all conditions are met:
    //  - the centroids are float16/float32 or uint8 with hamming distance
    //  - `num_centroids * dimension >= MIN_CENTROID_VALUES_FOR_INDEX`
    //  - the exact flat assignment is not already faster, see
    //    `flat_amx_assignment_blocker`
    pub fn may_train_index(
        centroids: ArrayRef,
        dimension: usize,
        distance_type: DistanceType,
    ) -> Result<Option<Self>> {
        // Guarded rather than divided by: every branch below needs the centroid
        // count, and a zero-width vector has none to index. Declining leaves the
        // caller on the flat path, which reports the malformed shape properly;
        // dividing here would panic before it got the chance.
        let Some(num_centroids) = centroids.len().checked_div(dimension) else {
            warn!("IVF partition assignment: no centroid index for a zero dimension");
            return Ok(None);
        };
        let decision = assignment_route(
            centroids.data_type(),
            num_centroids,
            dimension,
            distance_type,
            AmxAvailability::current(),
            &USE_HNSW_SPEEDUP_INDEXING,
        );
        // Once per k-means training call -- a few dozen lines over a build, since
        // hierarchical k-means retrains per level. That is the right granularity
        // for `debug!`: the matching gate in `compute_membership_and_dist` runs
        // per block of vectors, where the same line would flood.
        debug!(
            "IVF partition assignment: route={}, centroid_type={}, \
             num_centroids={num_centroids}, dimension={dimension}, \
             distance_type={distance_type}, amx={}, reason={}",
            decision.route,
            centroids.data_type(),
            decision.amx_blocker.unwrap_or("in use"),
            decision.reason,
        );
        if !decision.train_index {
            return Ok(None);
        }

        let store = match (centroids.data_type(), distance_type) {
            (DataType::Float16 | DataType::Float32 | DataType::Float64, _) => {
                let fsl = FixedSizeListArray::try_new_from_values(centroids, dimension as i32)?;
                SimpleStore::Float(FlatFloatStorage::new(fsl, distance_type))
            }
            (DataType::UInt8, DistanceType::Hamming) => {
                let fsl = FixedSizeListArray::try_new_from_values(centroids, dimension as i32)?;
                SimpleStore::Binary(FlatBinStorage::new(fsl, distance_type))
            }
            _ => return Ok(None),
        };
        Self::try_new(store).map(Some)
    }

    pub(crate) fn search(&self, query: ArrayRef) -> Result<(u32, f32)> {
        let params = HnswQueryParams {
            ef: APPROX_ASSIGN.ef,
            lower_bound: None,
            upper_bound: None,
            dist_q_c: 0.0,
            use_acorn: false,
        };
        let res = match &self.store {
            SimpleStore::Float(store) => self.index.search_basic(query, 1, &params, None, store)?,
            SimpleStore::Binary(store) => {
                let query = if query.data_type() == &DataType::UInt8 {
                    query
                } else {
                    cast(&query, &DataType::UInt8).map_err(|e| Error::index(e.to_string()))?
                };
                self.index.search_basic(query, 1, &params, None, store)?
            }
        };
        Ok((res[0].id, res[0].dist.0))
    }
}

#[inline]
pub(crate) fn do_prefetch<T>(ptrs: Range<*const T>) {
    // TODO use rust intrinsics instead of x86 intrinsics
    // TODO finish this
    unsafe {
        let (ptr, end_ptr) = (ptrs.start as *const i8, ptrs.end as *const i8);
        let mut current_ptr = ptr;
        while current_ptr < end_ptr {
            const CACHE_LINE_SIZE: usize = 64;
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                use core::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
                _mm_prefetch(current_ptr, _MM_HINT_T0);
            }
            current_ptr = current_ptr.add(CACHE_LINE_SIZE);
        }
    }
}

impl From<pb::tensor::DataType> for DataType {
    fn from(dt: pb::tensor::DataType) -> Self {
        match dt {
            pb::tensor::DataType::Uint8 => Self::UInt8,
            pb::tensor::DataType::Uint16 => Self::UInt16,
            pb::tensor::DataType::Uint32 => Self::UInt32,
            pb::tensor::DataType::Uint64 => Self::UInt64,
            pb::tensor::DataType::Float16 => Self::Float16,
            pb::tensor::DataType::Float32 => Self::Float32,
            pb::tensor::DataType::Float64 => Self::Float64,
            pb::tensor::DataType::Bfloat16 => unimplemented!(),
        }
    }
}

impl TryFrom<&DataType> for pb::tensor::DataType {
    type Error = Error;

    fn try_from(dt: &DataType) -> Result<Self> {
        match dt {
            DataType::UInt8 => Ok(Self::Uint8),
            DataType::UInt16 => Ok(Self::Uint16),
            DataType::UInt32 => Ok(Self::Uint32),
            DataType::UInt64 => Ok(Self::Uint64),
            DataType::Float16 => Ok(Self::Float16),
            DataType::Float32 => Ok(Self::Float32),
            DataType::Float64 => Ok(Self::Float64),
            _ => Err(Error::index(format!(
                "pb tensor type not supported: {:?}",
                dt
            ))),
        }
    }
}

impl TryFrom<DataType> for pb::tensor::DataType {
    type Error = Error;

    fn try_from(dt: DataType) -> Result<Self> {
        (&dt).try_into()
    }
}

impl TryFrom<&FixedSizeListArray> for pb::Tensor {
    type Error = Error;

    fn try_from(array: &FixedSizeListArray) -> Result<Self> {
        let mut tensor = Self::default();
        tensor.data_type = pb::tensor::DataType::try_from(array.value_type())? as i32;
        tensor.shape = vec![Array::len(array) as u32, array.value_length() as u32];
        let flat_array = array.values();
        tensor.data = flat_array.into_data().buffers()[0].to_vec();
        Ok(tensor)
    }
}

impl TryFrom<&pb::Tensor> for FixedSizeListArray {
    type Error = Error;

    fn try_from(tensor: &Tensor) -> Result<Self> {
        if tensor.shape.len() != 2 {
            return Err(Error::index(format!(
                "only accept 2-D tensor shape, got: {:?}",
                tensor.shape
            )));
        }
        let dim = tensor.shape[1] as usize;
        let num_rows = tensor.shape[0] as usize;
        let num_values = dim.checked_mul(num_rows).ok_or_else(|| {
            Error::index(format!(
                "Tensor shape {:?} exceeds the supported size",
                tensor.shape
            ))
        })?;
        let data_type = DataType::from(pb::tensor::DataType::try_from(tensor.data_type).unwrap());
        let expected_data_len =
            num_values
                .checked_mul(data_type.byte_width())
                .ok_or_else(|| {
                    Error::index(format!(
                        "Tensor shape {:?} exceeds the supported byte length",
                        tensor.shape
                    ))
                })?;
        if tensor.data.len() != expected_data_len {
            return Err(Error::index(format!(
                "Tensor shape {:?} with data type {data_type} requires {expected_data_len} bytes, got {}",
                tensor.shape,
                tensor.data.len()
            )));
        }

        let buffer = Buffer::from_bytes_bytes(
            bytes::Bytes::from(tensor.data.clone()),
            data_type.byte_width() as u64,
        );
        let data = ArrayData::builder(data_type)
            .len(num_values)
            .null_count(0)
            .add_buffer(buffer)
            .build()?;
        let flat_array = make_array(data);
        let field = Field::new("item", flat_array.data_type().clone(), true);
        Ok(Self::try_new(
            Arc::new(field),
            dim as i32,
            flat_array,
            None,
        )?)
    }
}

/// Check if all vectors in the FixedSizeListArray are finite
/// null values are considered as not finite
/// returns a BooleanArray
/// with the same length as the FixedSizeListArray
/// with true for finite values and false for non-finite values
pub fn is_finite(fsl: &FixedSizeListArray) -> BooleanArray {
    let is_finite = fsl
        .iter()
        .map(|v| match v {
            Some(v) => match v.data_type() {
                DataType::Float16 => {
                    let v = v.as_primitive::<Float16Type>();
                    Array::null_count(v) == 0 && v.values().iter().all(|v| v.is_finite())
                }
                DataType::Float32 => {
                    let v = v.as_primitive::<Float32Type>();
                    Array::null_count(v) == 0 && v.values().iter().all(|v| v.is_finite())
                }
                DataType::Float64 => {
                    let v = v.as_primitive::<Float64Type>();
                    Array::null_count(v) == 0 && v.values().iter().all(|v| v.is_finite())
                }
                _ => Array::null_count(&v) == 0,
            },
            None => false,
        })
        .collect::<Vec<_>>();
    BooleanArray::from(is_finite)
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow_array::{Float16Array, Float32Array, Float64Array, UInt8Array};
    use half::f16;
    use lance_arrow::FixedSizeListArrayExt;
    use num_traits::identities::Zero;
    use rayon::ThreadPoolBuilder;

    use arrow::compute::cast;
    use rstest::rstest;

    fn build_index(centroids: ArrayRef, dim: usize) -> SimpleIndex {
        let f32_centroids = cast(&centroids, &DataType::Float32).unwrap();
        let fsl = FixedSizeListArray::try_new_from_values(f32_centroids, dim as i32).unwrap();
        let store = SimpleStore::Float(FlatFloatStorage::new(fsl, DistanceType::L2));
        SimpleIndex::try_new(store).unwrap()
    }

    fn build_binary_index(centroids: ArrayRef, dim: usize) -> SimpleIndex {
        let u8_centroids = if centroids.data_type() == &DataType::UInt8 {
            centroids
        } else {
            cast(&centroids, &DataType::UInt8).unwrap()
        };
        let fsl = FixedSizeListArray::try_new_from_values(u8_centroids, dim as i32).unwrap();
        let store = SimpleStore::Binary(FlatBinStorage::new(fsl, DistanceType::Hamming));
        SimpleIndex::try_new(store).unwrap()
    }

    #[rstest]
    #[case::f16(Arc::new(Float16Array::from(
        (0..100).flat_map(|i| std::iter::repeat_n(f16::from_f32(i as f32), 16)).collect::<Vec<_>>(),
    )) as ArrayRef, 42.0f32)]
    #[case::f32(Arc::new(Float32Array::from(
        (0..100).flat_map(|i| std::iter::repeat_n(i as f32, 16)).collect::<Vec<_>>(),
    )) as ArrayRef, 42.0f32)]
    fn test_simple_index_nearest_centroid(#[case] centroids: ArrayRef, #[case] query_val: f32) {
        let thread_pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let index = thread_pool.install(|| build_index(centroids, 16));
        let query: ArrayRef = Arc::new(Float32Array::from(vec![query_val; 16]));
        let (id, dist) = index.search(query).unwrap();
        assert_eq!(id, 42);
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_simple_index_nearest_centroid_binary() {
        let centroids: ArrayRef = Arc::new(UInt8Array::from(
            (0..100)
                .flat_map(|i| std::iter::repeat_n(i as u8, 16))
                .collect::<Vec<_>>(),
        ));
        let index = build_binary_index(centroids, 16);
        let query: ArrayRef = Arc::new(UInt8Array::from(vec![42u8; 16]));
        let (id, dist) = index.search(query).unwrap();
        assert_eq!(id, 42);
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_simple_index_rejects_f64() {
        let centroids: ArrayRef = Arc::new(Float64Array::from(vec![0.0; 1600]));
        let result = SimpleIndex::may_train_index(centroids, 16, DistanceType::L2).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_simple_index_rejects_uint8_non_hamming() {
        let centroids: ArrayRef = Arc::new(UInt8Array::from(vec![0u8; 1600]));
        let result = SimpleIndex::may_train_index(centroids, 16, DistanceType::L2).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_fsl_to_tensor() {
        let fsl =
            FixedSizeListArray::try_new_from_values(Float16Array::from(vec![f16::zero(); 20]), 5)
                .unwrap();
        let tensor = pb::Tensor::try_from(&fsl).unwrap();
        assert_eq!(tensor.data_type, pb::tensor::DataType::Float16 as i32);
        assert_eq!(tensor.shape, vec![4, 5]);
        assert_eq!(tensor.data.len(), 20 * 2);
        let decoded = FixedSizeListArray::try_from(&tensor).unwrap();
        assert_eq!(decoded.values().to_data(), fsl.values().to_data());

        let fsl =
            FixedSizeListArray::try_new_from_values(Float32Array::from(vec![0.0; 20]), 5).unwrap();
        let tensor = pb::Tensor::try_from(&fsl).unwrap();
        assert_eq!(tensor.data_type, pb::tensor::DataType::Float32 as i32);
        assert_eq!(tensor.shape, vec![4, 5]);
        assert_eq!(tensor.data.len(), 20 * 4);
        let decoded = FixedSizeListArray::try_from(&tensor).unwrap();
        assert_eq!(decoded.values().to_data(), fsl.values().to_data());

        let fsl =
            FixedSizeListArray::try_new_from_values(Float64Array::from(vec![0.0; 20]), 5).unwrap();
        let tensor = pb::Tensor::try_from(&fsl).unwrap();
        assert_eq!(tensor.data_type, pb::tensor::DataType::Float64 as i32);
        assert_eq!(tensor.shape, vec![4, 5]);
        assert_eq!(tensor.data.len(), 20 * 8);
        let decoded = FixedSizeListArray::try_from(&tensor).unwrap();
        assert_eq!(decoded.values().to_data(), fsl.values().to_data());
    }

    #[rstest]
    #[case::too_short(vec![0; 7])]
    #[case::too_long(vec![0; 9])]
    fn test_tensor_to_fsl_rejects_invalid_data_length(#[case] data: Vec<u8>) {
        let tensor = pb::Tensor {
            data_type: pb::tensor::DataType::Uint32 as i32,
            shape: vec![1, 2],
            data,
        };

        let error = FixedSizeListArray::try_from(&tensor).unwrap_err();
        assert!(error.to_string().contains("requires 8 bytes"));
    }

    /// Every shape the AMX-FP16 GEMM cannot serve must keep the centroid index,
    /// because without the GEMM the flat path it would fall back to is ~2.7x
    /// slower than the index. These are the exact complement of the gate in
    /// `compute_membership_and_dist`.
    ///
    /// The reported reason is asserted too, not just that there was one: it is
    /// the only signal an A/B run has for which path a build took, so a reason
    /// that names the wrong condition is as bad as no reason at all.
    #[rstest]
    #[case::not_a_float_the_kernel_takes(&DataType::Float64, 10_000, 768, DistanceType::Dot, "centroid type")]
    #[case::not_dot(&DataType::Float16, 10_000, 768, DistanceType::L2, "distance type")]
    #[case::not_dot_f32(&DataType::Float32, 10_000, 768, DistanceType::L2, "distance type")]
    #[case::dim_below_one_k_pass(&DataType::Float16, 10_000, 31, DistanceType::Dot, "dimension")]
    #[case::k_below_one_b_block(&DataType::Float16, 31, 768, DistanceType::Dot, "centroid count")]
    #[case::dim_below_one_k_pass_f32(&DataType::Float32, 10_000, 31, DistanceType::Dot, "dimension")]
    #[case::k_below_one_b_block_f32(&DataType::Float32, 31, 768, DistanceType::Dot, "centroid count")]
    fn test_flat_amx_assignment_declines_unsupported_shapes(
        #[case] centroid_type: &DataType,
        #[case] num_centroids: usize,
        #[case] dimension: usize,
        #[case] distance_type: DistanceType,
        #[case] expected_reason: &str,
    ) {
        // Every route in service, so the shape is the only thing left to decline
        // on. Availability is a parameter precisely so this holds on any host.
        let blocker = flat_amx_assignment_blocker(
            centroid_type,
            num_centroids,
            dimension,
            distance_type,
            ALL_ROUTES,
        )
        .expect("this shape must be declined");
        assert!(
            blocker.contains(expected_reason),
            "blocked for {blocker:?}, expected something about {expected_reason:?}"
        );
    }

    /// Both routes in service — the state a Granite Rapids host with neither
    /// kill switch set is in, and the one where only the *shape* can decline.
    const ALL_ROUTES: AmxAvailability = AmxAvailability {
        native: true,
        rounded: true,
    };

    /// The blocker's dependence on the two switches, enumerated.
    ///
    /// Availability is a parameter rather than a read of the environment so this
    /// can be a table. Derived from `amx_fp32_cast_available()` instead, the fp32
    /// row would assert that a function returns what the very call it makes
    /// returns — true even if fp32 were wired to the f16 switch, which is the
    /// mistake that would send an `LANCE_AMX_FP32_CAST=0` build down the exact
    /// path with no GEMM under it.
    #[rstest]
    #[case::everything_on(true, true, None, None)]
    #[case::cast_off(true, false, None, Some("LANCE_AMX_FP32_CAST is off"))]
    #[case::amx_off(
        false,
        false,
        Some("AMX-FP16 is unavailable"),
        Some("AMX-FP16 is unavailable")
    )]
    fn test_flat_amx_assignment_follows_amx_availability(
        #[case] native: bool,
        #[case] rounded: bool,
        #[case] f16_blocker: Option<&str>,
        #[case] f32_blocker: Option<&str>,
    ) {
        let amx = AmxAvailability { native, rounded };
        for (centroid_type, want) in [
            (&DataType::Float16, f16_blocker),
            (&DataType::Float32, f32_blocker),
        ] {
            let got =
                flat_amx_assignment_blocker(centroid_type, 10_000, 768, DistanceType::Dot, amx);
            match want {
                None => assert_eq!(got, None, "{centroid_type} with {amx:?}"),
                Some(want) => assert!(
                    got.is_some_and(|reason| reason.contains(want)),
                    "{centroid_type} with {amx:?}: blocked for {got:?}, expected {want:?}"
                ),
            }
        }
    }

    /// Production reads the switches through exactly one function, and this is
    /// the assertion that it is still the right one. Derived rather than
    /// enumerated because that is all a `LazyLock`-cached switch permits — the
    /// table above is where the routing itself is pinned.
    #[test]
    fn test_amx_availability_reads_both_switches() {
        let current = AmxAvailability::current();
        assert_eq!(current.native, amx_fp16_available());
        assert_eq!(current.rounded, amx_fp32_cast_available());
    }

    /// What the `debug!` line reports, over the routes it has to keep apart —
    /// including the one an fp32 A/B run switches between.
    ///
    /// That line is the only signal separating the two arms: the k-means loss is
    /// expected to be identical to four decimal places with the cast on and off,
    /// which is the result the cast exists to produce, so the loss cannot
    /// confirm which path a build took. If the line ever reported the same route
    /// for both arms, an A/B run would silently compare a path against itself.
    ///
    /// Availability is enumerated, not read from the process. With the cast off,
    /// an exact-capable shape stops being exact-capable — so `train_index` flips
    /// to `true` and the route to `hnsw`. Hardcoding "must not build an index"
    /// here is what made the `LANCE_AMX_FP32_CAST=0` and `LANCE_DISABLE_AMX=1`
    /// runs of this suite fail, and would fail CI on any host without AMX.
    #[rstest]
    #[case::big_f16_auto(&DataType::Float16, 10_000, 768, SimpleIndexStatus::Auto, false)]
    #[case::big_f32_auto(&DataType::Float32, 10_000, 768, SimpleIndexStatus::Auto, false)]
    #[case::small_f32_auto(&DataType::Float32, 100, 1536, SimpleIndexStatus::Auto, true)]
    #[case::big_f16_disabled(&DataType::Float16, 10_000, 768, SimpleIndexStatus::Disabled, false)]
    fn test_assignment_route_names_the_path_with_amx(
        #[case] centroid_type: &DataType,
        #[case] num_centroids: usize,
        #[case] dimension: usize,
        #[case] status: SimpleIndexStatus,
        #[case] below_index_threshold: bool,
    ) {
        for amx in [
            ALL_ROUTES,
            AmxAvailability {
                native: true,
                rounded: false,
            },
            AmxAvailability {
                native: false,
                rounded: false,
            },
        ] {
            let decision = assignment_route(
                centroid_type,
                num_centroids,
                dimension,
                DistanceType::Dot,
                amx,
                &status,
            );
            let usable = flat_amx_assignment_blocker(
                centroid_type,
                num_centroids,
                dimension,
                DistanceType::Dot,
                amx,
            )
            .is_none();
            let context = format!("{centroid_type} with {amx:?}");
            // An index is built exactly when the GEMM cannot serve the shape and
            // nothing has already settled the question -- below the index
            // threshold, or with the index switched off, assignment stays flat
            // however unusable the GEMM is.
            let decided_elsewhere =
                below_index_threshold || matches!(status, SimpleIndexStatus::Disabled);
            let train_index = !usable && !decided_elsewhere;

            assert_eq!(decision.amx_blocker.is_none(), usable, "{context}");
            assert_eq!(
                decision.train_index, train_index,
                "{context}: route={} reason={}",
                decision.route, decision.reason
            );
            assert_eq!(
                decision.route,
                match (train_index, usable) {
                    (true, _) => "hnsw",
                    (false, true) => "flat+amx",
                    (false, false) => "flat",
                },
                "{context}"
            );
            if !train_index {
                assert_eq!(
                    below_index_threshold,
                    decision.reason.contains("below the size"),
                    "{context}: reason={}",
                    decision.reason
                );
            }
        }
    }

    /// A shape the GEMM cannot serve keeps the approximate index, and the line
    /// has to name the condition that sent it there rather than just "hnsw".
    #[test]
    fn test_assignment_route_reports_why_it_went_approximate() {
        let decision = assignment_route(
            &DataType::Float16,
            10_000,
            768,
            DistanceType::L2,
            ALL_ROUTES,
            &SimpleIndexStatus::Auto,
        );
        assert!(decision.train_index);
        assert_eq!(decision.route, "hnsw");
        assert!(
            decision.reason.contains("distance type"),
            "reason={}",
            decision.reason
        );
        assert_eq!(decision.amx_blocker, Some(decision.reason));
    }

    /// Below the index threshold the route is flat either way, so `reason` says
    /// the same thing with the cast on and off. `amx_blocker` is then the only
    /// field that separates the two arms of an fp32 A/B run — the case this
    /// pins, and the shape (100 x 1536) a first local run actually has.
    #[test]
    fn test_assignment_route_reports_the_amx_blocker_on_a_small_centroid_set() {
        let cast_off = AmxAvailability {
            native: true,
            rounded: false,
        };
        let on = assignment_route(
            &DataType::Float32,
            100,
            1536,
            DistanceType::Dot,
            ALL_ROUTES,
            &SimpleIndexStatus::Auto,
        );
        let off = assignment_route(
            &DataType::Float32,
            100,
            1536,
            DistanceType::Dot,
            cast_off,
            &SimpleIndexStatus::Auto,
        );

        assert!(!on.train_index && !off.train_index);
        assert_eq!(
            on.reason, off.reason,
            "the threshold reason cannot tell the arms apart -- that is the point"
        );
        assert_eq!(on.route, "flat+amx");
        assert_eq!(off.route, "flat");
        assert_eq!(on.amx_blocker, None);
        assert_eq!(off.amx_blocker, Some("LANCE_AMX_FP32_CAST is off"));
    }
}
