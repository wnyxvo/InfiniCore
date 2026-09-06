#ifndef __PAGED_CACHING_KERNEL_CUH__
#define __PAGED_CACHING_KERNEL_CUH__

//================================================================================
// Paged Caching Operator CUDA Kernel
//
// This kernel implements the "paged_caching" operation, which copies Key and Value
// vectors from a contiguous source tensor into a paged, non-contiguous KV Cache.
//
// Design Principles:
// 1. Token/head parallelism: a 2D grid maps x to KV head and y to token;
//    each CUDA block copies one token for one KV head.
// 2. Scalar strided copy: each thread covers elements `threadIdx.x + n*NUM_THREADS`;
//    no vectorized instruction is assumed by this implementation.
// 3. Layout preservation: source and cache strides are supplied by the descriptor,
//    so padding and non-contiguous supported layouts retain their slot semantics.
//================================================================================

namespace op::paged_caching::cuda {

template <
    typename Tdata, // Data type of the tensors (e.g., half, __nv_bfloat16)
    int NUM_THREADS, // Number of threads per block, configured at launch time
    bool VECTORIZED = false, // 8-byte path for four 16-bit values
    int HEADS_PER_CTA = 1 // grouped mapping: one warp per KV head
    >
__device__ void pagedCachingKernel(
    // ----- Output Tensors -----
    Tdata *k_cache_ptr, // Pointer to the destination K cache pool [num_blocks, nkvh, block_size, dh]
    Tdata *v_cache_ptr, // Pointer to the destination V cache pool [num_blocks, nkvh, block_size, dh]
    // ----- Input Tensors -----
    const Tdata *k_ptr,              // Pointer to the source Keys, shape [ntok, nkvh, dh]
    const Tdata *v_ptr,              // Pointer to the source Values, shape [ntok, nkvh, dh]
    const int64_t *slot_mapping_ptr, // Pointer to the slot mapping, shape [ntok]
    // ----- Metadata -----
    const size_t head_size,   // Dimension of each key head (dh_k)
    const size_t v_head_size, // Dimension of each value head (dh_v)
    const size_t block_size,  // Number of tokens per block in the KV cache
    const size_t num_kv_heads,
    // ----- Stride Information -----
    const ptrdiff_t k_src_stride,         // Stride between tokens in the source K tensor
    const ptrdiff_t v_src_stride,         // Stride between tokens in the source V tensor
    const ptrdiff_t k_src_head_stride,    // Stride between heads in the source K tensor
    const ptrdiff_t v_src_head_stride,    // Stride between heads in the source V tensor
    const ptrdiff_t k_src_size_stride,    // Stride between elements in a source K head
    const ptrdiff_t v_src_size_stride,    // Stride between elements in a source V head
    const ptrdiff_t k_cache_block_stride, // Stride between blocks in the K cache pool
    const ptrdiff_t v_cache_block_stride, // Stride between blocks in the V cache pool
    const ptrdiff_t k_cache_head_stride,  // Stride between heads in the K cache pool
    const ptrdiff_t v_cache_head_stride,  // Stride between heads in the V cache pool
    const ptrdiff_t k_cache_slot_stride,  // Stride between block slots in the K cache pool
    const ptrdiff_t v_cache_slot_stride,  // Stride between block slots in the V cache pool
    const ptrdiff_t k_cache_size_stride, // Stride between elements in a K cache head
    const ptrdiff_t v_cache_size_stride  // Stride between elements in a V cache head
) {
    //================================================================================
    // 1. Identify Work Unit & Calculate Addresses
    //================================================================================

    // Each block processes one token.
    const int token_idx = blockIdx.y;
    // Single-head candidates retain their historical thread mapping (128/256
    // threads cooperate on one head). Grouped candidates use one warp per head.
    int head_idx;
    int lane;
    int stride;
    if constexpr (HEADS_PER_CTA == 1) {
        head_idx = blockIdx.x;
        lane = threadIdx.x;
        stride = NUM_THREADS;
    } else {
        const int warp_idx = threadIdx.x / 32;
        lane = threadIdx.x % 32;
        head_idx = blockIdx.x * HEADS_PER_CTA + warp_idx;
        stride = 32;
        if (head_idx >= static_cast<int>(num_kv_heads)) return;
    }

    // Retrieve the destination slot for the current token.
    const int64_t slot_idx = slot_mapping_ptr[token_idx];

    // Handle padding: if slot_idx is negative, this token is padding and should be ignored.
    if (slot_idx < 0) {
        return;
    }
    // Calculate the physical block index and the offset within that block.
    const int64_t physical_block_idx = slot_idx / block_size;
    const int64_t block_offset = slot_idx % block_size;

    // Calculate base pointers for source and destination for this specific token.
    const Tdata *k_src_head_ptr = k_ptr + token_idx * k_src_stride + head_idx * k_src_head_stride;
    const Tdata *v_src_head_ptr = v_ptr + token_idx * v_src_stride + head_idx * v_src_head_stride;

    // Destination pointer calculation assumes a [num_blocks, block_size, num_heads, head_size] layout.
    // We point to the beginning of the memory region for this token's slot.
    Tdata *k_cache_block_base_ptr = k_cache_ptr + physical_block_idx * k_cache_block_stride;
    Tdata *k_dst_head_ptr = k_cache_block_base_ptr + head_idx * k_cache_head_stride + block_offset * k_cache_slot_stride;

    Tdata *v_cache_block_base_ptr = v_cache_ptr + physical_block_idx * v_cache_block_stride;
    Tdata *v_dst_head_ptr = v_cache_block_base_ptr + head_idx * v_cache_head_stride + block_offset * v_cache_slot_stride;

    // Every address is expressed in elements so arbitrary outer strides and
    // storage offsets retain their existing semantics. The vector path is
    // entered only when the host has proved unit innermost strides and 8-byte
    // alignment for every token/head/slot address.
    if constexpr (VECTORIZED) {
        constexpr int ELEMENTS_PER_VECTOR = 4;
        for (int i = lane * ELEMENTS_PER_VECTOR; i < static_cast<int>(head_size); i += stride * ELEMENTS_PER_VECTOR) {
            const auto *src = reinterpret_cast<const uint2 *>(k_src_head_ptr + i * k_src_size_stride);
            auto *dst = reinterpret_cast<uint2 *>(k_dst_head_ptr + i * k_cache_size_stride);
            *dst = *src;
        }
        for (int i = lane * ELEMENTS_PER_VECTOR; i < static_cast<int>(v_head_size); i += stride * ELEMENTS_PER_VECTOR) {
            const auto *src = reinterpret_cast<const uint2 *>(v_src_head_ptr + i * v_src_size_stride);
            auto *dst = reinterpret_cast<uint2 *>(v_dst_head_ptr + i * v_cache_size_stride);
            *dst = *src;
        }
    } else {
        for (int i = lane; i < static_cast<int>(head_size); i += stride) {
            k_dst_head_ptr[i * k_cache_size_stride] = k_src_head_ptr[i * k_src_size_stride];
        }
        for (int i = lane; i < static_cast<int>(v_head_size); i += stride) {
            v_dst_head_ptr[i * v_cache_size_stride] = v_src_head_ptr[i * v_src_size_stride];
        }
    }
}

} // namespace op::paged_caching::cuda

#endif // __PAGED_CACHING_KERNEL_CUH__
