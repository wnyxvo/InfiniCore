#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../cuda/kernel.cuh"
#include "paged_caching_nvidia.cuh"
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>

template <typename Tdata, int NUM_THREADS, bool VECTORIZED = false, int HEADS_PER_CTA = 1>
INFINIOP_CUDA_KERNEL pagedCaching(
    Tdata *k_cache, Tdata *v_cache,
    const Tdata *k, const Tdata *v,
    const int64_t *slot_mapping,
    const size_t head_size, const size_t v_head_size, const size_t block_size,
    const size_t num_kv_heads,
    const ptrdiff_t k_src_stride, const ptrdiff_t v_src_stride,
    const ptrdiff_t k_src_head_stride, const ptrdiff_t v_src_head_stride,
    const ptrdiff_t k_src_size_stride, const ptrdiff_t v_src_size_stride,
    const ptrdiff_t k_cache_block_stride, const ptrdiff_t v_cache_block_stride,
    const ptrdiff_t k_cache_head_stride, const ptrdiff_t v_cache_head_stride,
    const ptrdiff_t k_cache_slot_stride, const ptrdiff_t v_cache_slot_stride,
    const ptrdiff_t k_cache_size_stride, const ptrdiff_t v_cache_size_stride) {
    op::paged_caching::cuda::pagedCachingKernel<Tdata, NUM_THREADS, VECTORIZED, HEADS_PER_CTA>(
        k_cache, v_cache, k, v, slot_mapping, head_size, v_head_size,
        block_size, num_kv_heads, k_src_stride, v_src_stride,
        k_src_head_stride, v_src_head_stride, k_src_size_stride, v_src_size_stride,
        k_cache_block_stride, v_cache_block_stride, k_cache_head_stride, v_cache_head_stride, k_cache_slot_stride, v_cache_slot_stride,
        k_cache_size_stride, v_cache_size_stride);
}

namespace op::paged_caching::nvidia {
// PIMPL struct definition
struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

// Destructor implementation
Descriptor::~Descriptor() {
    delete _opaque;
}

// Static factory method implementation
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t k_cache_desc,
    infiniopTensorDescriptor_t v_cache_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t slot_mapping_desc) {

    auto info = PagedCachingInfo::create(k_cache_desc, v_cache_desc, k_desc, v_desc, slot_mapping_desc);
    CHECK_RESULT(info);

    // Create and return the Descriptor instance.
    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info.take(), 0, handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

// The launchKernel function is a templated helper to encapsulate the CUDA kernel launch.
// It sets up grid/block dimensions and calls the device-side kernel.
template <int NUM_THREADS, bool VECTORIZED = false, int HEADS_PER_CTA = 1>
infiniStatus_t launchKernel(const PagedCachingInfo &info,
                            void *k_cache, void *v_cache,
                            infiniDtype_t dtype,
                            const void *k, const void *v,
                            const void *slot_mapping,
                            size_t num_tokens, size_t num_kv_heads, size_t head_size, size_t v_head_size, size_t block_size,
                            ptrdiff_t k_src_stride, ptrdiff_t v_src_stride,
                            ptrdiff_t k_src_head_stride, ptrdiff_t v_src_head_stride,
                            ptrdiff_t k_src_size_stride, ptrdiff_t v_src_size_stride,
                            ptrdiff_t k_cache_block_stride, ptrdiff_t v_cache_block_stride,
                            ptrdiff_t k_cache_head_stride, ptrdiff_t v_cache_head_stride,
                            ptrdiff_t k_cache_slot_stride, ptrdiff_t v_cache_slot_stride,
                            ptrdiff_t k_cache_size_stride, ptrdiff_t v_cache_size_stride,
                            cudaStream_t stream) {

    // The 2D grid maps x to head groups and y to tokens. A grouped CTA owns
    // HEADS_PER_CTA warps, with one independent warp per KV head.
    dim3 grid((uint64_t(num_kv_heads) + HEADS_PER_CTA - 1) / HEADS_PER_CTA, uint64_t(num_tokens), 1);
    // Block dimension is 1D, using the number of threads specified at compile time.
    dim3 block(NUM_THREADS);

    // This kernel does not require dynamic shared memory.
    size_t shared_mem_size = 0;

    // Launch the device-side CUDA kernel.
    if (dtype == INFINI_DTYPE_F16) {
        pagedCaching<half, NUM_THREADS, VECTORIZED, HEADS_PER_CTA>
            <<<grid, block, shared_mem_size, stream>>>(
                (half *)k_cache,
                (half *)v_cache,
                (const half *)k,
                (const half *)v,
                (const int64_t *)slot_mapping,
                head_size,
                v_head_size,
                block_size,
                num_kv_heads,
                k_src_stride,
                v_src_stride,
                k_src_head_stride,
                v_src_head_stride,
                k_src_size_stride,
                v_src_size_stride,
                k_cache_block_stride,
                v_cache_block_stride,
                k_cache_head_stride,
                v_cache_head_stride,
                k_cache_slot_stride,
                v_cache_slot_stride,
                k_cache_size_stride,
                v_cache_size_stride);
    } else if (dtype == INFINI_DTYPE_BF16) {
        pagedCaching<__nv_bfloat16, NUM_THREADS, VECTORIZED, HEADS_PER_CTA>
            <<<grid, block, shared_mem_size, stream>>>(
                (__nv_bfloat16 *)k_cache,
                (__nv_bfloat16 *)v_cache,
                (const __nv_bfloat16 *)k,
                (const __nv_bfloat16 *)v,
                (const int64_t *)slot_mapping,
                head_size,
                v_head_size,
                block_size,
                num_kv_heads,
                k_src_stride,
                v_src_stride,
                k_src_head_stride,
                v_src_head_stride,
                k_src_size_stride,
                v_src_size_stride,
                k_cache_block_stride,
                v_cache_block_stride,
                k_cache_head_stride,
                v_cache_head_stride,
                k_cache_slot_stride,
                v_cache_slot_stride,
                k_cache_size_stride,
                v_cache_size_stride);
    } else if (dtype == INFINI_DTYPE_F32) {
        pagedCaching<float, NUM_THREADS, VECTORIZED, HEADS_PER_CTA>
            <<<grid, block, shared_mem_size, stream>>>(
                (float *)k_cache,
                (float *)v_cache,
                (const float *)k,
                (const float *)v,
                (const int64_t *)slot_mapping,
                head_size,
                v_head_size,
                block_size,
                num_kv_heads,
                k_src_stride,
                v_src_stride,
                k_src_head_stride,
                v_src_head_stride,
                k_src_size_stride,
                v_src_size_stride,
                k_cache_block_stride,
                v_cache_block_stride,
                k_cache_head_stride,
                v_cache_head_stride,
                k_cache_slot_stride,
                v_cache_slot_stride,
                k_cache_size_stride,
                v_cache_size_stride);
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}

// Vector writes require every per-head base address to remain 8-byte aligned.
// The data pointer check includes storage_offset; stride checks cover all token,
// head, block and slot offsets reachable by the 2-D grid.
static bool vector_path_eligible(const PagedCachingInfo &info,
                                const void *k_cache, const void *v_cache,
                                const void *k, const void *v) {
    if (info.dtype != INFINI_DTYPE_F16 && info.dtype != INFINI_DTYPE_BF16) return false;
    if (info.head_size % 4 != 0 || info.v_head_size % 4 != 0) return false;
    if (info.k_src_size_stride != 1 || info.v_src_size_stride != 1 ||
        info.k_cache_size_stride != 1 || info.v_cache_size_stride != 1) return false;
    constexpr ptrdiff_t ELEMENT_BYTES = 2;
    auto stride_aligned = [](ptrdiff_t stride) {
        return (stride % (8 / ELEMENT_BYTES)) == 0;
    };
    const ptrdiff_t strides[] = {
        info.k_src_stride, info.v_src_stride,
        info.k_src_head_stride, info.v_src_head_stride,
        info.k_cache_block_stride, info.v_cache_block_stride,
        info.k_cache_head_stride, info.v_cache_head_stride,
        info.k_cache_slot_stride, info.v_cache_slot_stride};
    for (auto stride : strides) if (!stride_aligned(stride)) return false;
    const uintptr_t pointers[] = {
        reinterpret_cast<uintptr_t>(k_cache), reinterpret_cast<uintptr_t>(v_cache),
        reinterpret_cast<uintptr_t>(k), reinterpret_cast<uintptr_t>(v)};
    for (auto pointer : pointers) if ((pointer & 7U) != 0) return false;
    return true;
}

// Execution method implementation
infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *k_cache, void *v_cache,
    const void *k, const void *v,
    const void *slot_mapping,
    void *stream_) const {

    cudaStream_t stream = (cudaStream_t)stream_;

    // Default remains 1024 threads. An explicit test-only override allows
    // comparing existing template instantiations without changing grid/layout.
    int requested_threads = 0;
    if (const char *env = std::getenv("INFINIOP_PAGED_CACHING_THREADS")) {
        requested_threads = std::atoi(env);
    }
    const int max_threads = _opaque->internal->maxThreadsPerBlock();
    const char *vector_env = std::getenv("INFINIOP_PAGED_CACHING_VECTOR");
    const bool vector_requested = vector_env &&
        (std::strcmp(vector_env, "1") == 0 || std::strcmp(vector_env, "true") == 0);
    const bool vector_eligible = vector_path_eligible(_info, k_cache, v_cache, k, v);
    int requested_heads_per_cta = 1;
    if (const char *env = std::getenv("INFINIOP_PAGED_CACHING_HEADS_PER_CTA")) {
        const int value = std::atoi(env);
        if (value == 2 || value == 4) requested_heads_per_cta = value;
    }
    const bool grouped_e = requested_threads == 64 && requested_heads_per_cta == 2 && max_threads >= 64;
    const bool grouped_f = requested_threads == 128 && requested_heads_per_cta == 4 && max_threads >= 128;
    const bool grouped = grouped_e || grouped_f;
    const bool use_vector = vector_requested && vector_eligible &&
        ((requested_threads == 32 && requested_heads_per_cta == 1) || grouped);
    const int selected_threads = grouped_e ? 64 : (grouped_f ? 128 :
        (requested_threads == 32 ? 32 :
        (requested_threads == 128 ? 128 : (requested_threads == 256 ? 256 :
         (max_threads >= CUDA_BLOCK_SIZE_1024 ? CUDA_BLOCK_SIZE_1024 :
          (max_threads >= CUDA_BLOCK_SIZE_512 ? CUDA_BLOCK_SIZE_512 : CUDA_BLOCK_SIZE_4096))))));
    const int selected_heads_per_cta = grouped ? requested_heads_per_cta : 1;
    if (std::getenv("INFINIOP_PAGED_CACHING_DEBUG")) {
        std::fprintf(stderr, "[paged_caching] selected_threads=%d heads_per_cta=%d grid_x=%zu num_heads=%zu vector_requested=%d vector_eligible=%d vector_active=%d dtype=%d\n",
                     selected_threads, selected_heads_per_cta,
                     (_info.num_kv_heads + selected_heads_per_cta - 1) / selected_heads_per_cta, _info.num_kv_heads,
                     static_cast<int>(vector_requested), static_cast<int>(vector_eligible),
                     static_cast<int>(use_vector), static_cast<int>(_info.dtype));
    }
    if (grouped_e) {
        if (use_vector) {
            launchKernel<64, true, 2>(
                _info, k_cache, v_cache, _info.dtype, k, v, slot_mapping,
                _info.num_tokens, _info.num_kv_heads, _info.head_size, _info.v_head_size, _info.block_size,
                _info.k_src_stride, _info.v_src_stride, _info.k_src_head_stride, _info.v_src_head_stride,
                _info.k_src_size_stride, _info.v_src_size_stride,
                _info.k_cache_block_stride, _info.v_cache_block_stride,
                _info.k_cache_head_stride, _info.v_cache_head_stride,
                _info.k_cache_slot_stride, _info.v_cache_slot_stride,
                _info.k_cache_size_stride, _info.v_cache_size_stride, stream);
        } else {
            launchKernel<64, false, 2>(
                _info, k_cache, v_cache, _info.dtype, k, v, slot_mapping,
                _info.num_tokens, _info.num_kv_heads, _info.head_size, _info.v_head_size, _info.block_size,
                _info.k_src_stride, _info.v_src_stride, _info.k_src_head_stride, _info.v_src_head_stride,
                _info.k_src_size_stride, _info.v_src_size_stride,
                _info.k_cache_block_stride, _info.v_cache_block_stride,
                _info.k_cache_head_stride, _info.v_cache_head_stride,
                _info.k_cache_slot_stride, _info.v_cache_slot_stride,
                _info.k_cache_size_stride, _info.v_cache_size_stride, stream);
        }
    } else if (grouped_f) {
        if (use_vector) {
            launchKernel<128, true, 4>(
                _info, k_cache, v_cache, _info.dtype, k, v, slot_mapping,
                _info.num_tokens, _info.num_kv_heads, _info.head_size, _info.v_head_size, _info.block_size,
                _info.k_src_stride, _info.v_src_stride, _info.k_src_head_stride, _info.v_src_head_stride,
                _info.k_src_size_stride, _info.v_src_size_stride,
                _info.k_cache_block_stride, _info.v_cache_block_stride,
                _info.k_cache_head_stride, _info.v_cache_head_stride,
                _info.k_cache_slot_stride, _info.v_cache_slot_stride,
                _info.k_cache_size_stride, _info.v_cache_size_stride, stream);
        } else {
            launchKernel<128, false, 4>(
                _info, k_cache, v_cache, _info.dtype, k, v, slot_mapping,
                _info.num_tokens, _info.num_kv_heads, _info.head_size, _info.v_head_size, _info.block_size,
                _info.k_src_stride, _info.v_src_stride, _info.k_src_head_stride, _info.v_src_head_stride,
                _info.k_src_size_stride, _info.v_src_size_stride,
                _info.k_cache_block_stride, _info.v_cache_block_stride,
                _info.k_cache_head_stride, _info.v_cache_head_stride,
                _info.k_cache_slot_stride, _info.v_cache_slot_stride,
                _info.k_cache_size_stride, _info.v_cache_size_stride, stream);
        }
    } else if (requested_threads == 32 && max_threads >= 32) {
        if (use_vector) {
            launchKernel<32, true>(
                _info, k_cache, v_cache, _info.dtype, k, v, slot_mapping,
                _info.num_tokens, _info.num_kv_heads, _info.head_size, _info.v_head_size, _info.block_size,
                _info.k_src_stride, _info.v_src_stride, _info.k_src_head_stride, _info.v_src_head_stride,
                _info.k_src_size_stride, _info.v_src_size_stride,
                _info.k_cache_block_stride, _info.v_cache_block_stride,
                _info.k_cache_head_stride, _info.v_cache_head_stride,
                _info.k_cache_slot_stride, _info.v_cache_slot_stride,
                _info.k_cache_size_stride, _info.v_cache_size_stride, stream);
        } else {
            launchKernel<32, false>(
                _info, k_cache, v_cache, _info.dtype, k, v, slot_mapping,
                _info.num_tokens, _info.num_kv_heads, _info.head_size, _info.v_head_size, _info.block_size,
                _info.k_src_stride, _info.v_src_stride, _info.k_src_head_stride, _info.v_src_head_stride,
                _info.k_src_size_stride, _info.v_src_size_stride,
                _info.k_cache_block_stride, _info.v_cache_block_stride,
                _info.k_cache_head_stride, _info.v_cache_head_stride,
                _info.k_cache_slot_stride, _info.v_cache_slot_stride,
                _info.k_cache_size_stride, _info.v_cache_size_stride, stream);
        }
    } else if (requested_threads == 128 && max_threads >= 128) {
        launchKernel<128>(
            _info, k_cache, v_cache, _info.dtype, k, v, slot_mapping,
            _info.num_tokens, _info.num_kv_heads, _info.head_size, _info.v_head_size, _info.block_size,
            _info.k_src_stride, _info.v_src_stride, _info.k_src_head_stride, _info.v_src_head_stride,
            _info.k_src_size_stride, _info.v_src_size_stride,
            _info.k_cache_block_stride, _info.v_cache_block_stride, _info.k_cache_head_stride, _info.v_cache_head_stride,
            _info.k_cache_slot_stride, _info.v_cache_slot_stride,
                _info.k_cache_size_stride, _info.v_cache_size_stride, stream);
    } else if (requested_threads == 256 && max_threads >= 256) {
        launchKernel<256>(
            _info, k_cache, v_cache, _info.dtype, k, v, slot_mapping,
            _info.num_tokens, _info.num_kv_heads, _info.head_size, _info.v_head_size, _info.block_size,
            _info.k_src_stride, _info.v_src_stride, _info.k_src_head_stride, _info.v_src_head_stride,
            _info.k_src_size_stride, _info.v_src_size_stride,
            _info.k_cache_block_stride, _info.v_cache_block_stride, _info.k_cache_head_stride, _info.v_cache_head_stride,
            _info.k_cache_slot_stride, _info.v_cache_slot_stride,
                _info.k_cache_size_stride, _info.v_cache_size_stride, stream);
    } else if (max_threads >= CUDA_BLOCK_SIZE_1024) {
        // Dispatch based on data type for a 1024-thread block.
        launchKernel<CUDA_BLOCK_SIZE_1024>(
            _info, k_cache, v_cache, _info.dtype, k, v, slot_mapping,
            _info.num_tokens, _info.num_kv_heads, _info.head_size, _info.v_head_size, _info.block_size,
            _info.k_src_stride, _info.v_src_stride,
            _info.k_src_head_stride, _info.v_src_head_stride,
            _info.k_src_size_stride, _info.v_src_size_stride,
            _info.k_cache_block_stride, _info.v_cache_block_stride,
            _info.k_cache_head_stride, _info.v_cache_head_stride,
            _info.k_cache_slot_stride, _info.v_cache_slot_stride,
            _info.k_cache_size_stride, _info.v_cache_size_stride,
            stream);
    } else if (_opaque->internal->maxThreadsPerBlock() >= CUDA_BLOCK_SIZE_512) {
        launchKernel<CUDA_BLOCK_SIZE_512>(
            _info, k_cache, v_cache, _info.dtype, k, v, slot_mapping,
            _info.num_tokens, _info.num_kv_heads, _info.head_size, _info.v_head_size, _info.block_size,
            _info.k_src_stride, _info.v_src_stride,
            _info.k_src_head_stride, _info.v_src_head_stride,
            _info.k_src_size_stride, _info.v_src_size_stride,
            _info.k_cache_block_stride, _info.v_cache_block_stride,
            _info.k_cache_head_stride, _info.v_cache_head_stride,
            _info.k_cache_slot_stride, _info.v_cache_slot_stride,
            _info.k_cache_size_stride, _info.v_cache_size_stride,
            stream);
    } else if (_opaque->internal->maxThreadsPerBlock() >= CUDA_BLOCK_SIZE_4096) {
        launchKernel<CUDA_BLOCK_SIZE_4096>(
            _info, k_cache, v_cache, _info.dtype, k, v, slot_mapping,
            _info.num_tokens, _info.num_kv_heads, _info.head_size, _info.v_head_size, _info.block_size,
            _info.k_src_stride, _info.v_src_stride,
            _info.k_src_head_stride, _info.v_src_head_stride,
            _info.k_src_size_stride, _info.v_src_size_stride,
            _info.k_cache_block_stride, _info.v_cache_block_stride,
            _info.k_cache_head_stride, _info.v_cache_head_stride,
            _info.k_cache_slot_stride, _info.v_cache_slot_stride,
            _info.k_cache_size_stride, _info.v_cache_size_stride,
            stream);
    } else {
        // If the GPU is older and supports fewer threads, return an error.
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::paged_caching::nvidia
