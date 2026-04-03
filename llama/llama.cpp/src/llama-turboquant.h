// SPDX-License-Identifier: MIT
// TurboQuant KV cache compression for llama.cpp
// Phase 1: baseline CPU quantize/dequantize with F16 staging

#pragma once

#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

struct ggml_tensor;

namespace turboquant {

// Sidecar metadata: Pi, S, centroids loaded from .tqmeta file
struct meta {
    double   bits         = 3.0;
    int      head_dim     = 128;
    int      k_centroids  = 0;
    int      codebook_id  = 0;  // 0=paper, 1=ternary
    double   qjl_factor   = 0.0;

    bool     hadamard     = false;   // true = use WHT rotation (v3 sidecar)

    std::vector<float> centroids;  // [k_centroids]
    std::vector<float> pi;         // [head_dim * head_dim] row-major (dense mode)
    std::vector<float> s;          // [head_dim * head_dim] row-major (dense mode)

    // Hadamard mode: packed sign bits instead of dense matrices
    std::vector<uint8_t> signs_pi; // [head_dim/8] bit-packed
    std::vector<uint8_t> signs_s;  // [head_dim/8] bit-packed
};

// Byte layout for one packed head-row
struct row_layout {
    int      idx_bits_per_coord = 0;   // ceil(log2(k_centroids))
    uint32_t idx_bytes     = 0;   // packed centroid indices
    uint32_t norm_bytes    = 4;   // float32 x_norm
    uint32_t sign_bytes    = 0;   // bit-packed QJL signs
    uint32_t gamma_bytes   = 4;   // float32 gamma

    // Offsets within one head-row
    uint32_t norm_offset   = 0;
    uint32_t sign_offset   = 0;
    uint32_t gamma_offset  = 0;
    uint32_t total_bytes   = 0;
};

// Shadow buffer for one layer
struct layer_buffer {
    std::vector<uint8_t> k_packed;  // [kv_size * n_kv_heads * layout.total_bytes]
    std::vector<uint8_t> v_packed;
};

// Full TurboQuant state
struct state {
    bool     enabled    = false;
    meta     m;
    row_layout layout;
    std::vector<layer_buffer> layers;

    int      n_kv_heads = 0;
    int      head_dim   = 0;
    uint32_t kv_size    = 0;
};

// Load .tqmeta sidecar file
meta load_meta(const std::string & path);

// Compute packed row layout from metadata
row_layout compute_row_layout(const meta & m);

// Quantize one head-row: src_f32[head_dim] -> dst_packed[layout.total_bytes]
void quantize_head_row(
    const meta       & m,
    const row_layout & layout,
    const float      * src_f32,    // [head_dim]
    uint8_t          * dst_packed  // [layout.total_bytes]
);

// Dequantize one head-row: src_packed[layout.total_bytes] -> dst_f32[head_dim]
void dequant_head_row(
    const meta       & m,
    const row_layout & layout,
    const uint8_t    * src_packed,  // [layout.total_bytes]
    float            * dst_f32     // [head_dim]
);

// Post-process: for a single KV row (all heads), quantize then dequantize in-place
// This reads F32 from the cache, compresses to shadow buffer, decompresses back
void post_process_row(
    state        & st,
    ggml_tensor  * k_tensor,   // [n_embd_k_gqa, kv_size, ...]
    ggml_tensor  * v_tensor,
    uint32_t       row_idx,    // which KV slot
    uint32_t       layer_idx,
    int            n_kv_heads,
    int            head_dim
);

// Initialize state: load meta, compute layout, allocate shadow buffers
std::unique_ptr<state> init(
    const std::string & tqmeta_path,
    uint32_t kv_size,
    int n_layers,
    int n_kv_heads,
    int head_dim
);

// Register dequantize/quantize function pointers in the GGML type traits table
// Must be called after loading the sidecar (needs Pi/S/centroids state)
void register_ggml_type(const state & st);

} // namespace turboquant
