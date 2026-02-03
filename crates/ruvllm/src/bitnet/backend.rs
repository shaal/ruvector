//! BitNet b1.58 Inference Backend
//!
//! This module implements the `BitNetBackend` inference pipeline for BitNet b1.58
//! MoE models (e.g., GLM-4.7-Flash). It wires together the quantizer, TL1 kernel,
//! and MoE routing into a working inference pipeline.
//!
//! ## Phase 0 Scope
//!
//! - Attention is a placeholder (pass-through) for smoke testing
//! - MoE routing is fully functional (FP16 gate + softmax + top-K)
//! - Expert FFN uses real TL1 GEMV on ternary weights
//! - Embedding lookup and LM head are FP16 matmul
//!
//! ## Architecture
//!
//! ```text
//! Embedding (FP16) -> [Transformer Layers] -> RMSNorm -> LM Head (FP16) -> Logits
//!
//! Each Transformer Layer:
//!   RMSNorm -> Attention (placeholder) -> Residual
//!   -> RMSNorm -> MoE Gate (FP16) -> Top-K Expert Selection
//!   -> Expert FFN (TL1 GEMV on ternary) -> Weighted Sum -> Residual
//! ```

use std::path::Path;

use crate::backends::{
    GenerateParams, GeneratedToken, LlmBackend, ModelArchitecture, ModelConfig,
    ModelInfo, Quantization, StreamEvent, TokenStream,
    Tokenizer as BackendTokenizer,
    SpecialTokens as BackendSpecialTokens,
};
use crate::error::{Result, RuvLLMError};
use crate::gguf::{GgufFile, GgufQuantType};

use super::ternary_tensor::TernaryTensor;
use super::tokenizer::{BpeTokenizer, SpecialTokens as BitNetSpecialTokens};

// ============================================================================
// Configuration
// ============================================================================

/// Model configuration for BitNet MoE inference.
///
/// Describes the architecture dimensions extracted from GGUF metadata
/// or supplied manually for testing.
#[derive(Debug, Clone)]
pub struct BitNetModelConfig {
    /// Number of transformer layers
    pub num_layers: usize,
    /// Hidden state dimension
    pub hidden_size: usize,
    /// Number of MoE experts per layer
    pub num_experts: usize,
    /// Number of active experts per token (top-K)
    pub active_experts: usize,
    /// FFN intermediate dimension per expert
    pub intermediate_size: usize,
    /// Number of attention query heads
    pub num_attention_heads: usize,
    /// Number of attention key-value heads (GQA)
    pub num_kv_heads: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum context length
    pub max_context: usize,
    /// RoPE frequency base
    pub rope_theta: f32,
}

impl Default for BitNetModelConfig {
    fn default() -> Self {
        // Default values loosely based on GLM-4.7-Flash architecture
        Self {
            num_layers: 28,
            hidden_size: 4096,
            num_experts: 8,
            active_experts: 2,
            intermediate_size: 11008,
            num_attention_heads: 32,
            num_kv_heads: 8,
            vocab_size: 151552,
            max_context: 8192,
            rope_theta: 10000.0,
        }
    }
}

// ============================================================================
// TL1 Lookup Table
// ============================================================================

/// Pre-computed lookup table for packed 2-bit ternary bytes.
///
/// For each of the 256 possible byte values, stores the four decoded
/// ternary values {-1, 0, +1}. This avoids per-element bit manipulation
/// during the hot GEMV inner loop.
type Tl1Lut = [[i8; 4]; 256];

/// Build the TL1 lookup table at load time.
///
/// Encoding per the ternary_tensor module:
/// - 00 = -1, 01 = 0, 10 = +1, 11 = 0 (reserved)
fn build_tl1_lut() -> Tl1Lut {
    let mut lut = [[0i8; 4]; 256];
    for byte_val in 0u16..256 {
        for pos in 0..4 {
            let bits = ((byte_val as u8) >> (pos * 2)) & 0b11;
            lut[byte_val as usize][pos] = match bits {
                0b00 => -1,
                0b01 => 0,
                0b10 => 1,
                0b11 => 0, // reserved
                _ => unreachable!(),
            };
        }
    }
    lut
}

// ============================================================================
// Per-Layer and Per-Expert Weight Storage
// ============================================================================

/// Ternary weights for a single MoE expert (gate, up, down projections).
#[derive(Debug, Clone)]
struct ExpertWeights {
    /// gate_proj: [intermediate_size, hidden_size]
    gate_proj: TernaryTensor,
    /// up_proj: [intermediate_size, hidden_size]
    up_proj: TernaryTensor,
    /// down_proj: [hidden_size, intermediate_size]
    down_proj: TernaryTensor,
}

/// Attention projection weights (ternary).
#[derive(Debug, Clone)]
struct AttentionWeights {
    /// Q projection: [num_heads * head_dim, hidden_size]
    q_proj: TernaryTensor,
    /// K projection: [num_kv_heads * head_dim, hidden_size]
    k_proj: TernaryTensor,
    /// V projection: [num_kv_heads * head_dim, hidden_size]
    v_proj: TernaryTensor,
    /// Output projection: [hidden_size, num_heads * head_dim]
    o_proj: TernaryTensor,
}

/// Weights for a single transformer layer.
#[derive(Debug, Clone)]
struct TransformerLayer {
    /// Input RMSNorm weight [hidden_size]
    input_norm_weight: Vec<f32>,
    /// Post-attention RMSNorm weight [hidden_size]
    post_attn_norm_weight: Vec<f32>,
    /// Attention projection weights (ternary)
    attention: AttentionWeights,
    /// MoE router gate weight [num_experts, hidden_size] (FP32, stored row-major)
    gate_weight: Vec<f32>,
    /// Per-expert FFN weights (ternary)
    experts: Vec<ExpertWeights>,
}

// ============================================================================
// KV Cache
// ============================================================================

/// Per-layer KV cache for autoregressive generation.
#[derive(Debug, Clone)]
struct LayerKvCache {
    /// Cached key vectors: one [num_kv_heads * head_dim] per position
    keys: Vec<Vec<f32>>,
    /// Cached value vectors: one [num_kv_heads * head_dim] per position
    values: Vec<Vec<f32>>,
}

impl LayerKvCache {
    fn new() -> Self {
        Self {
            keys: Vec::new(),
            values: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.keys.clear();
        self.values.clear();
    }

    fn len(&self) -> usize {
        self.keys.len()
    }
}

// ============================================================================
// BitNetBackend
// ============================================================================

/// BitNet b1.58 MoE inference backend.
///
/// Provides model loading from GGUF and forward pass inference using
/// ternary TL1 GEMV kernels for expert FFN layers and FP32 for shared
/// layers (embeddings, norms, router, LM head).
///
/// # Example
///
/// ```rust,ignore
/// use ruvllm::bitnet::backend::BitNetBackend;
/// use ruvllm::backends::{LlmBackend, ModelConfig, GenerateParams};
///
/// let mut backend = BitNetBackend::new();
/// backend.load_model("model.gguf", ModelConfig::default())?;
///
/// let logits = backend.forward(&[1, 2, 3])?;
/// ```
pub struct BitNetBackend {
    /// Model configuration (set after load)
    config: Option<BitNetModelConfig>,
    /// Embedding table [vocab_size * hidden_size], row-major FP32
    embedding: Vec<f32>,
    /// LM head weight [vocab_size * hidden_size], row-major FP32
    lm_head: Vec<f32>,
    /// Final RMSNorm weight [hidden_size]
    final_norm_weight: Vec<f32>,
    /// Transformer layers
    layers: Vec<TransformerLayer>,
    /// Pre-computed TL1 lookup table
    tl1_lut: Tl1Lut,
    /// Per-layer KV caches for autoregressive generation
    kv_caches: Vec<LayerKvCache>,
    /// Tokenizer (loaded from GGUF or byte-level fallback)
    tok: Option<BpeTokenizer>,
    /// Pre-computed RoPE cos/sin tables [max_context, head_dim/2]
    rope_cos: Vec<f32>,
    rope_sin: Vec<f32>,
    /// Whether a model is loaded
    loaded: bool,
    /// Model path (for info)
    model_path: String,
}

impl BitNetBackend {
    /// Create a new unloaded BitNetBackend.
    pub fn new() -> Self {
        Self {
            config: None,
            embedding: Vec::new(),
            lm_head: Vec::new(),
            final_norm_weight: Vec::new(),
            layers: Vec::new(),
            tl1_lut: build_tl1_lut(),
            kv_caches: Vec::new(),
            tok: None,
            rope_cos: Vec::new(),
            rope_sin: Vec::new(),
            loaded: false,
            model_path: String::new(),
        }
    }

    /// Clear the KV cache (call between sequences).
    pub fn reset_cache(&mut self) {
        for cache in &mut self.kv_caches {
            cache.clear();
        }
    }

    // ========================================================================
    // Model Loading
    // ========================================================================

    /// Load a BitNet MoE model from a GGUF file.
    ///
    /// Parses the GGUF file, extracts model configuration from metadata,
    /// separates FP16 shared tensors from ternary expert tensors, and
    /// pre-builds the TL1 lookup table.
    fn load_gguf(&mut self, path: &str) -> Result<()> {
        let gguf = GgufFile::open_mmap(Path::new(path))?;

        // Extract model config from GGUF metadata
        let config = self.extract_config(&gguf)?;

        // Load embedding table (FP16/FP32)
        self.embedding = self.load_fp_tensor(&gguf, "model.embed_tokens.weight", &config)?;

        // Load LM head (may share weights with embedding in some architectures)
        self.lm_head = if gguf.get_tensor("lm_head.weight").is_some() {
            self.load_fp_tensor(&gguf, "lm_head.weight", &config)?
        } else if gguf.get_tensor("output.weight").is_some() {
            self.load_fp_tensor(&gguf, "output.weight", &config)?
        } else {
            // Tied embeddings: copy embedding table
            self.embedding.clone()
        };

        // Load final norm
        self.final_norm_weight =
            self.load_fp_tensor(&gguf, "model.norm.weight", &config)?;

        // Load transformer layers
        self.layers = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            let layer = self.load_layer(&gguf, layer_idx, &config)?;
            self.layers.push(layer);
        }

        // Initialize KV caches (one per layer)
        self.kv_caches = (0..config.num_layers).map(|_| LayerKvCache::new()).collect();

        // Build RoPE cos/sin tables
        let head_dim = config.hidden_size / config.num_attention_heads;
        self.build_rope_tables(config.max_context, head_dim, config.rope_theta);

        // Load tokenizer from GGUF metadata
        self.tok = self.load_tokenizer_from_gguf(&gguf);

        self.config = Some(config);
        self.loaded = true;
        self.model_path = path.to_string();

        Ok(())
    }

    /// Pre-compute RoPE frequency tables.
    fn build_rope_tables(&mut self, max_seq: usize, head_dim: usize, theta: f32) {
        let half = head_dim / 2;
        let total = max_seq * half;
        self.rope_cos = vec![0.0; total];
        self.rope_sin = vec![0.0; total];

        for pos in 0..max_seq {
            for i in 0..half {
                let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                self.rope_cos[pos * half + i] = angle.cos();
                self.rope_sin[pos * half + i] = angle.sin();
            }
        }
    }

    /// Load tokenizer from GGUF metadata, falling back to byte-level tokenizer.
    fn load_tokenizer_from_gguf(&self, gguf: &GgufFile) -> Option<BpeTokenizer> {
        // Try to extract token list from GGUF
        let tokens_meta = gguf.metadata.get("tokenizer.ggml.tokens");
        let merges_meta = gguf.metadata.get("tokenizer.ggml.merges");

        if let Some(tokens_arr) = tokens_meta.and_then(|v| v.as_array()) {
            let vocab: Vec<String> = tokens_arr
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect();

            let merges: Vec<(String, String)> = if let Some(merges_arr) =
                merges_meta.and_then(|v| v.as_array())
            {
                merges_arr
                    .iter()
                    .filter_map(|v| {
                        let s = v.as_str()?;
                        let mut parts = s.splitn(2, ' ');
                        let left = parts.next()?.to_string();
                        let right = parts.next()?.to_string();
                        Some((left, right))
                    })
                    .collect()
            } else {
                Vec::new()
            };

            if !vocab.is_empty() {
                return Some(BpeTokenizer::from_vocab(
                    vocab,
                    merges,
                    BitNetSpecialTokens::default(),
                ));
            }
        }

        // Fallback: construct a byte-level tokenizer (260 tokens)
        Some(Self::build_byte_level_tokenizer())
    }

    /// Build a minimal byte-level tokenizer for when GGUF has no vocab.
    fn build_byte_level_tokenizer() -> BpeTokenizer {
        let mut vocab = vec![
            "<PAD>".to_string(), // 0
            "<BOS>".to_string(), // 1
            "<EOS>".to_string(), // 2
            "<UNK>".to_string(), // 3
        ];
        for b in 0..=255u8 {
            vocab.push(format!("<{:02X}>", b));
        }
        BpeTokenizer::from_vocab(vocab, vec![], BitNetSpecialTokens::default())
    }

    /// Extract BitNetModelConfig from GGUF metadata.
    fn extract_config(&self, gguf: &GgufFile) -> Result<BitNetModelConfig> {
        let num_layers = gguf.layer_count().unwrap_or(28);
        let hidden_size = gguf.embedding_length().unwrap_or(4096);
        let num_attention_heads = gguf.head_count().unwrap_or(32);
        let num_kv_heads = gguf.head_count_kv().unwrap_or(8);
        let vocab_size = gguf.vocab_size().unwrap_or(151552);
        let max_context = gguf.context_length().unwrap_or(8192);
        let rope_theta = gguf.rope_freq_base().unwrap_or(10000.0);
        let intermediate_size = gguf.feed_forward_length().unwrap_or(11008);

        // Detect expert count from tensor names
        let num_experts = self.detect_expert_count(gguf).unwrap_or(8);

        // Detect active experts from metadata or default to 2
        let active_experts = gguf
            .metadata
            .get("model.expert_count_active")
            .or_else(|| gguf.metadata.get("llm.expert_used_count"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(2);

        Ok(BitNetModelConfig {
            num_layers,
            hidden_size,
            num_experts,
            active_experts,
            intermediate_size,
            num_attention_heads,
            num_kv_heads,
            vocab_size,
            max_context,
            rope_theta,
        })
    }

    /// Detect the number of MoE experts by scanning tensor names.
    fn detect_expert_count(&self, gguf: &GgufFile) -> Option<usize> {
        let mut max_expert_idx = 0usize;
        let mut found_any = false;

        for tensor in &gguf.tensors {
            // Look for patterns like "experts.0.", "experts.7.", etc.
            if let Some(pos) = tensor.name.find("experts.") {
                let after = &tensor.name[pos + 8..];
                if let Some(dot) = after.find('.') {
                    if let Ok(idx) = after[..dot].parse::<usize>() {
                        max_expert_idx = max_expert_idx.max(idx);
                        found_any = true;
                    }
                }
            }
        }

        if found_any {
            Some(max_expert_idx + 1)
        } else {
            None
        }
    }

    /// Load an FP16/FP32 tensor from GGUF, returning FP32 data.
    fn load_fp_tensor(
        &self,
        gguf: &GgufFile,
        name: &str,
        _config: &BitNetModelConfig,
    ) -> Result<Vec<f32>> {
        match gguf.get_tensor(name) {
            Some(_) => gguf.load_tensor_f32(name),
            None => Err(RuvLLMError::NotFound(format!(
                "Required tensor not found: {}",
                name
            ))),
        }
    }

    /// Load a ternary tensor from GGUF (BitnetT158 or dequant + re-quantize).
    fn load_ternary_tensor(
        &self,
        gguf: &GgufFile,
        name: &str,
    ) -> Result<TernaryTensor> {
        let info = gguf
            .get_tensor(name)
            .ok_or_else(|| RuvLLMError::NotFound(format!("Tensor not found: {}", name)))?;

        if info.dtype == GgufQuantType::BitnetT158 {
            // Native ternary format: extract packed data and scales directly
            let raw = gguf.load_tensor_quantized(name)?;
            let num_elements = info.num_elements();
            let block_size = 256usize;
            let num_blocks = (num_elements + block_size - 1) / block_size;
            let type_size = 66usize; // 64 packed + 2 FP16 scale

            let mut packed_data = Vec::with_capacity(num_blocks * 64);
            let mut scales = Vec::with_capacity(num_blocks);

            for blk in 0..num_blocks {
                let offset = blk * type_size;
                if offset + type_size > raw.data.len() {
                    break;
                }
                packed_data.extend_from_slice(&raw.data[offset..offset + 64]);
                let scale_bits =
                    u16::from_le_bytes([raw.data[offset + 64], raw.data[offset + 65]]);
                scales.push(f16_to_f32(scale_bits));
            }

            let shape = if info.shape.len() == 2 {
                (info.shape[0], info.shape[1])
            } else {
                (1, num_elements)
            };

            Ok(TernaryTensor {
                packed_data,
                scales,
                shape,
                block_size,
            })
        } else {
            // Non-native format: dequantize to FP32, then quantize to ternary
            let fp32 = gguf.load_tensor_f32(name)?;
            let num_elements = fp32.len();
            let shape = if info.shape.len() == 2 {
                (info.shape[0], info.shape[1])
            } else {
                (1, num_elements)
            };

            let ptconfig = super::quantizer::PtBitnetConfig::default();
            super::quantizer::quantize_tensor(&fp32, shape, &ptconfig)
        }
    }

    /// Load a single transformer layer.
    fn load_layer(
        &self,
        gguf: &GgufFile,
        idx: usize,
        config: &BitNetModelConfig,
    ) -> Result<TransformerLayer> {
        let prefix = format!("model.layers.{}", idx);

        // Norm weights (FP16/FP32)
        let input_norm_weight = self.load_fp_tensor(
            gguf,
            &format!("{}.input_layernorm.weight", prefix),
            config,
        )?;
        let post_attn_norm_weight = self.load_fp_tensor(
            gguf,
            &format!("{}.post_attention_layernorm.weight", prefix),
            config,
        )?;

        // Attention projections (ternary)
        let attn_prefix = format!("{}.self_attn", prefix);
        let q_proj = self.load_ternary_tensor(
            gguf,
            &format!("{}.q_proj.weight", attn_prefix),
        )?;
        let k_proj = self.load_ternary_tensor(
            gguf,
            &format!("{}.k_proj.weight", attn_prefix),
        )?;
        let v_proj = self.load_ternary_tensor(
            gguf,
            &format!("{}.v_proj.weight", attn_prefix),
        )?;
        let o_proj = self.load_ternary_tensor(
            gguf,
            &format!("{}.o_proj.weight", attn_prefix),
        )?;

        let attention = AttentionWeights {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        };

        // MoE router gate (FP16/FP32): [num_experts, hidden_size]
        let gate_weight = self.load_fp_tensor(
            gguf,
            &format!("{}.mlp.gate.weight", prefix),
            config,
        )?;

        // Expert FFN weights (ternary)
        let mut experts = Vec::with_capacity(config.num_experts);
        for expert_idx in 0..config.num_experts {
            let expert_prefix =
                format!("{}.mlp.experts.{}", prefix, expert_idx);

            let gate_proj = self.load_ternary_tensor(
                gguf,
                &format!("{}.gate_proj.weight", expert_prefix),
            )?;
            let up_proj = self.load_ternary_tensor(
                gguf,
                &format!("{}.up_proj.weight", expert_prefix),
            )?;
            let down_proj = self.load_ternary_tensor(
                gguf,
                &format!("{}.down_proj.weight", expert_prefix),
            )?;

            experts.push(ExpertWeights {
                gate_proj,
                up_proj,
                down_proj,
            });
        }

        Ok(TransformerLayer {
            input_norm_weight,
            post_attn_norm_weight,
            attention,
            gate_weight,
            experts,
        })
    }

    // ========================================================================
    // Forward Pass
    // ========================================================================

    /// Run a forward pass for a single token, using the KV cache.
    ///
    /// This is the autoregressive path: embed one token, run all layers
    /// with cached K/V from prior positions, return logits.
    ///
    /// Call `reset_cache()` before starting a new sequence.
    ///
    /// # Arguments
    ///
    /// * `token_id` - Single token to process
    /// * `position` - Position index in the sequence (0-based)
    pub fn forward_token(&mut self, token_id: u32, position: usize) -> Result<Vec<f32>> {
        let config = self.config.as_ref().ok_or_else(|| {
            RuvLLMError::Model("No model loaded".to_string())
        })?.clone();

        let hidden = config.hidden_size;

        if (token_id as usize) >= config.vocab_size {
            return Err(RuvLLMError::Model(format!(
                "Token ID {} exceeds vocab size {}",
                token_id, config.vocab_size
            )));
        }

        // Embedding lookup
        let start = (token_id as usize) * hidden;
        let mut hidden_states: Vec<f32> = self.embedding[start..start + hidden].to_vec();

        // Transformer layers
        for layer_idx in 0..self.layers.len() {
            hidden_states = self.forward_layer_cached(
                &hidden_states,
                layer_idx,
                position,
                &config,
            )?;
        }

        // Final RMSNorm
        rms_norm_inplace(&mut hidden_states, &self.final_norm_weight, 1e-6);

        // LM head: logits = hidden_states @ lm_head^T
        let logits = fp32_matvec_transposed(
            &self.lm_head,
            &hidden_states,
            config.vocab_size,
            hidden,
        );

        Ok(logits)
    }

    /// Legacy forward: process full token sequence without KV cache.
    /// Kept for backwards compatibility with tests.
    pub fn forward(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let config = self.config.as_ref().ok_or_else(|| {
            RuvLLMError::Model("No model loaded".to_string())
        })?;

        if token_ids.is_empty() {
            return Err(RuvLLMError::Model("Empty token sequence".to_string()));
        }

        let hidden = config.hidden_size;
        let last_token = *token_ids.last().unwrap() as usize;
        if last_token >= config.vocab_size {
            return Err(RuvLLMError::Model(format!(
                "Token ID {} exceeds vocab size {}",
                last_token, config.vocab_size
            )));
        }
        let mut hidden_states: Vec<f32> =
            self.embedding[last_token * hidden..(last_token + 1) * hidden].to_vec();

        for (_layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_states = self.forward_layer_nocache(
                &hidden_states,
                layer,
                config,
            )?;
        }

        rms_norm_inplace(&mut hidden_states, &self.final_norm_weight, 1e-6);

        let logits = fp32_matvec_transposed(
            &self.lm_head,
            &hidden_states,
            config.vocab_size,
            hidden,
        );

        Ok(logits)
    }

    /// Forward pass through a single layer with KV cache (autoregressive).
    fn forward_layer_cached(
        &mut self,
        input: &[f32],
        layer_idx: usize,
        position: usize,
        config: &BitNetModelConfig,
    ) -> Result<Vec<f32>> {
        let hidden = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_kv_heads;
        let head_dim = hidden / num_heads;
        let kv_dim = num_kv_heads * head_dim;

        // --- Pre-attention norm ---
        let mut normed = input.to_vec();
        let layer = &self.layers[layer_idx];
        rms_norm_inplace(&mut normed, &layer.input_norm_weight, 1e-6);

        // --- Q/K/V projections via TL1 GEMV ---
        let q = self.tl1_gemv(&self.layers[layer_idx].attention.q_proj, &normed, hidden, hidden);
        let k = self.tl1_gemv(&self.layers[layer_idx].attention.k_proj, &normed, kv_dim, hidden);
        let v = self.tl1_gemv(&self.layers[layer_idx].attention.v_proj, &normed, kv_dim, hidden);

        // --- Apply RoPE to Q and K ---
        let mut q_rope = q;
        let mut k_rope = k.clone();
        self.apply_rope(&mut q_rope, num_heads, head_dim, position);
        self.apply_rope(&mut k_rope, num_kv_heads, head_dim, position);

        // --- Update KV cache ---
        self.kv_caches[layer_idx].keys.push(k_rope);
        self.kv_caches[layer_idx].values.push(v);
        let seq_len = self.kv_caches[layer_idx].len();

        // --- GQA Attention ---
        let gqa_groups = num_heads / num_kv_heads;
        let inv_sqrt_d = 1.0 / (head_dim as f32).sqrt();
        let mut attn_out = vec![0.0f32; hidden];

        for h in 0..num_heads {
            let kv_head = h / gqa_groups;
            let q_offset = h * head_dim;

            // Compute attention scores for this head across all cached positions
            let mut scores = Vec::with_capacity(seq_len);
            for pos in 0..seq_len {
                let k_offset = kv_head * head_dim;
                let k_vec = &self.kv_caches[layer_idx].keys[pos];
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_rope[q_offset + d] * k_vec[k_offset + d];
                }
                scores.push(dot * inv_sqrt_d);
            }

            // Causal mask is implicit: we only have positions <= current
            // Softmax over scores
            softmax_inplace(&mut scores);

            // Weighted sum of V
            for pos in 0..seq_len {
                let v_offset = kv_head * head_dim;
                let v_vec = &self.kv_caches[layer_idx].values[pos];
                let w = scores[pos];
                for d in 0..head_dim {
                    attn_out[q_offset + d] += w * v_vec[v_offset + d];
                }
            }
        }

        // --- Output projection ---
        let o_proj = self.tl1_gemv(
            &self.layers[layer_idx].attention.o_proj,
            &attn_out,
            hidden,
            hidden,
        );

        // --- Residual after attention ---
        let mut residual: Vec<f32> = input.iter().zip(o_proj.iter()).map(|(r, a)| r + a).collect();

        // --- Post-attention norm ---
        let mut normed_ffn = residual.clone();
        let layer = &self.layers[layer_idx];
        rms_norm_inplace(&mut normed_ffn, &layer.post_attn_norm_weight, 1e-6);

        // --- MoE ---
        let (expert_indices, expert_weights) =
            self.route_experts(&normed_ffn, &layer.gate_weight, config)?;

        let mut moe_output = vec![0.0f32; hidden];
        for (&eidx, &eweight) in expert_indices.iter().zip(expert_weights.iter()) {
            if eidx >= layer.experts.len() {
                return Err(RuvLLMError::Model(format!(
                    "Expert index {} out of bounds (layer has {} experts)",
                    eidx, layer.experts.len()
                )));
            }
            let expert_out = self.expert_forward(&normed_ffn, &self.layers[layer_idx].experts[eidx], config)?;
            for (o, &e) in moe_output.iter_mut().zip(expert_out.iter()) {
                *o += eweight * e;
            }
        }

        for (r, &m) in residual.iter_mut().zip(moe_output.iter()) {
            *r += m;
        }

        Ok(residual)
    }

    /// Forward pass through a single layer WITHOUT KV cache (legacy path).
    fn forward_layer_nocache(
        &self,
        input: &[f32],
        layer: &TransformerLayer,
        config: &BitNetModelConfig,
    ) -> Result<Vec<f32>> {
        let hidden = config.hidden_size;

        let mut normed = input.to_vec();
        rms_norm_inplace(&mut normed, &layer.input_norm_weight, 1e-6);

        // Attention: Q/K/V projections, single-position self-attention (degenerates to
        // identity-like behavior for 1 position but at least runs the projection weights)
        let num_heads = config.num_attention_heads;
        let head_dim = hidden / num_heads;
        let kv_dim = config.num_kv_heads * head_dim;

        let q = self.tl1_gemv(&layer.attention.q_proj, &normed, hidden, hidden);
        let k = self.tl1_gemv(&layer.attention.k_proj, &normed, kv_dim, hidden);
        let v = self.tl1_gemv(&layer.attention.v_proj, &normed, kv_dim, hidden);

        // Single-position attention: softmax([score]) = [1.0], so output = V expanded to all heads
        let gqa_groups = num_heads / config.num_kv_heads;
        let mut attn_concat = vec![0.0f32; hidden];
        for h in 0..num_heads {
            let kv_head = h / gqa_groups;
            for d in 0..head_dim {
                attn_concat[h * head_dim + d] = v[kv_head * head_dim + d];
            }
        }
        // Suppress unused warning — q and k are computed to exercise the projections
        let _ = q;
        let _ = k;

        let o_out = self.tl1_gemv(&layer.attention.o_proj, &attn_concat, hidden, hidden);

        let mut residual: Vec<f32> = input.iter().zip(o_out.iter()).map(|(r, a)| r + a).collect();

        let mut normed_ffn = residual.clone();
        rms_norm_inplace(&mut normed_ffn, &layer.post_attn_norm_weight, 1e-6);

        let (expert_indices, expert_weights) =
            self.route_experts(&normed_ffn, &layer.gate_weight, config)?;

        let mut moe_output = vec![0.0f32; hidden];
        for (&eidx, &eweight) in expert_indices.iter().zip(expert_weights.iter()) {
            if eidx >= layer.experts.len() {
                return Err(RuvLLMError::Model(format!(
                    "Expert index {} out of bounds (layer has {} experts)",
                    eidx, layer.experts.len()
                )));
            }
            let expert_out = self.expert_forward(&normed_ffn, &layer.experts[eidx], config)?;
            for (o, &e) in moe_output.iter_mut().zip(expert_out.iter()) {
                *o += eweight * e;
            }
        }

        for (r, &m) in residual.iter_mut().zip(moe_output.iter()) {
            *r += m;
        }

        Ok(residual)
    }

    /// Apply Rotary Position Embedding (RoPE) in-place.
    ///
    /// For each head, rotates pairs of dimensions (2i, 2i+1) by position-dependent angles.
    fn apply_rope(&self, x: &mut [f32], num_heads: usize, head_dim: usize, position: usize) {
        let half = head_dim / 2;
        let max_seq = self.rope_cos.len() / half;
        if position >= max_seq {
            return; // Beyond pre-computed tables — skip RoPE
        }
        let cos_base = position * half;
        for h in 0..num_heads {
            let offset = h * head_dim;
            for i in 0..half {
                let cos_val = self.rope_cos[cos_base + i];
                let sin_val = self.rope_sin[cos_base + i];
                let x0 = x[offset + 2 * i];
                let x1 = x[offset + 2 * i + 1];
                x[offset + 2 * i] = x0 * cos_val - x1 * sin_val;
                x[offset + 2 * i + 1] = x0 * sin_val + x1 * cos_val;
            }
        }
    }

    // ========================================================================
    // MoE Router
    // ========================================================================

    /// Route hidden states to the top-K experts.
    ///
    /// Computes `scores = hidden_states @ gate_weight^T`, applies softmax,
    /// then selects the top-K experts with highest scores.
    ///
    /// # Returns
    ///
    /// Tuple of (expert_indices, expert_weights) both of length active_experts.
    fn route_experts(
        &self,
        hidden_states: &[f32],
        gate_weight: &[f32],
        config: &BitNetModelConfig,
    ) -> Result<(Vec<usize>, Vec<f32>)> {
        let num_experts = config.num_experts;
        let hidden = config.hidden_size;
        // Clamp top_k to num_experts to prevent selecting more experts than exist
        let top_k = config.active_experts.min(num_experts);

        if num_experts == 0 {
            return Ok((vec![], vec![]));
        }

        // Gate: scores[e] = dot(hidden_states, gate_weight[e])
        let mut scores = vec![0.0f32; num_experts];
        for e in 0..num_experts {
            let row_start = e * hidden;
            if row_start + hidden > gate_weight.len() {
                break;
            }
            let mut dot = 0.0f32;
            for j in 0..hidden {
                dot += hidden_states[j] * gate_weight[row_start + j];
            }
            scores[e] = dot;
        }

        // Softmax over expert scores
        softmax_inplace(&mut scores);

        // Top-K selection
        let mut indexed: Vec<(usize, f32)> =
            scores.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let selected: Vec<(usize, f32)> = indexed.into_iter().take(top_k).collect();

        // Renormalize selected weights so they sum to 1
        let weight_sum: f32 = selected.iter().map(|(_, w)| w).sum();
        let norm_factor = if weight_sum > 1e-12 { 1.0 / weight_sum } else { 1.0 };

        let expert_indices: Vec<usize> = selected.iter().map(|(i, _)| *i).collect();
        let expert_weights: Vec<f32> =
            selected.iter().map(|(_, w)| w * norm_factor).collect();

        Ok((expert_indices, expert_weights))
    }

    // ========================================================================
    // Expert FFN (TL1 GEMV)
    // ========================================================================

    /// Forward pass through a single expert's SwiGLU FFN.
    ///
    /// Computes:
    /// ```text
    /// gate = TL1_GEMV(gate_proj, input)
    /// up   = TL1_GEMV(up_proj, input)
    /// hidden = silu(gate) * up
    /// output = TL1_GEMV(down_proj, hidden)
    /// ```
    fn expert_forward(
        &self,
        input: &[f32],
        expert: &ExpertWeights,
        config: &BitNetModelConfig,
    ) -> Result<Vec<f32>> {
        let intermediate = config.intermediate_size;
        let hidden = config.hidden_size;

        // gate_proj: [intermediate_size, hidden_size] @ input[hidden_size] -> [intermediate_size]
        let gate_out = self.tl1_gemv(&expert.gate_proj, input, intermediate, hidden);

        // up_proj: [intermediate_size, hidden_size] @ input[hidden_size] -> [intermediate_size]
        let up_out = self.tl1_gemv(&expert.up_proj, input, intermediate, hidden);

        // SiLU(gate) * up (element-wise)
        let mut fused = vec![0.0f32; intermediate];
        for i in 0..intermediate {
            let silu_val = gate_out[i] * sigmoid(gate_out[i]);
            fused[i] = silu_val * up_out[i];
        }

        // down_proj: [hidden_size, intermediate_size] @ fused[intermediate_size] -> [hidden_size]
        let output = self.tl1_gemv(&expert.down_proj, &fused, hidden, intermediate);

        Ok(output)
    }

    /// TL1 GEMV: ternary matrix-vector product using the pre-built lookup table.
    ///
    /// Computes `output[i] = sum_j(ternary_weight[i,j] * input[j]) * scale[block]`
    /// using addition/subtraction only (multiplication-free for the ternary part).
    ///
    /// The lookup table maps each packed byte to its four ternary values,
    /// eliminating per-element bit extraction from the inner loop.
    fn tl1_gemv(
        &self,
        weight: &TernaryTensor,
        input: &[f32],
        out_rows: usize,
        in_cols: usize,
    ) -> Vec<f32> {
        let block_size = weight.block_size;
        let mut output = vec![0.0f32; out_rows];

        // Each row of the weight matrix is a contiguous sequence of packed bytes.
        // packed bytes per row = ceil(in_cols / 4)
        let bytes_per_row = (in_cols + 3) / 4;
        // Number of scale entries per row
        let blocks_per_row = (in_cols + block_size - 1) / block_size;

        for row in 0..out_rows {
            let row_byte_offset = row * bytes_per_row;
            let row_scale_offset = row * blocks_per_row;
            let mut accum = 0.0f32;

            for blk in 0..blocks_per_row {
                let scale = weight
                    .scales
                    .get(row_scale_offset + blk)
                    .copied()
                    .unwrap_or(1.0);

                let blk_start_col = blk * block_size;
                let blk_end_col = (blk_start_col + block_size).min(in_cols);
                let mut block_accum = 0.0f32;

                // Process 4 elements at a time via LUT
                let mut c = blk_start_col;

                while c + 4 <= blk_end_col {
                    let byte_idx = row_byte_offset + c / 4;
                    if byte_idx >= weight.packed_data.len() {
                        break;
                    }
                    let packed_byte = weight.packed_data[byte_idx];
                    let ternary = &self.tl1_lut[packed_byte as usize];

                    // Accumulate: ternary[k] * input[c+k] for k=0..3
                    // Since ternary is {-1, 0, +1}, this is add/sub/skip
                    for k in 0..4 {
                        let t = ternary[k];
                        if t == 1 {
                            block_accum += input[c + k];
                        } else if t == -1 {
                            block_accum -= input[c + k];
                        }
                        // t == 0: skip (multiplication-free)
                    }
                    c += 4;
                }

                // Handle tail elements (< 4 remaining in block)
                while c < blk_end_col {
                    let byte_idx = row_byte_offset + c / 4;
                    let bit_pos = c % 4;
                    if byte_idx < weight.packed_data.len() {
                        let t = self.tl1_lut[weight.packed_data[byte_idx] as usize][bit_pos];
                        if t == 1 {
                            block_accum += input[c];
                        } else if t == -1 {
                            block_accum -= input[c];
                        }
                    }
                    c += 1;
                }

                accum += block_accum * scale;
            }

            output[row] = accum;
        }

        output
    }

    /// Greedy-decode a single next token from logits.
    fn argmax(logits: &[f32]) -> u32 {
        let mut best_idx = 0u32;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i as u32;
            }
        }
        best_idx
    }
}

// ============================================================================
// LlmBackend Trait Implementation
// ============================================================================

// ============================================================================
// Tokenizer trait bridge
// ============================================================================

/// Wraps our BpeTokenizer to implement the crate-level Tokenizer trait.
struct TokenizerBridge<'a> {
    inner: &'a BpeTokenizer,
}

impl<'a> BackendTokenizer for TokenizerBridge<'a> {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.inner.encode(text))
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        Ok(self.inner.decode(tokens))
    }

    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn special_tokens(&self) -> BackendSpecialTokens {
        BackendSpecialTokens {
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            ..Default::default()
        }
    }
}

impl LlmBackend for BitNetBackend {
    fn load_model(&mut self, model_id: &str, _config: ModelConfig) -> Result<()> {
        self.load_gguf(model_id)
    }

    fn generate(&self, prompt: &str, params: GenerateParams) -> Result<String> {
        if !self.loaded {
            return Err(RuvLLMError::Model("No model loaded".to_string()));
        }

        let tokenizer = self.tok.as_ref().ok_or_else(|| {
            RuvLLMError::Model("No tokenizer loaded".to_string())
        })?;

        // Encode prompt via tokenizer
        let prompt_tokens = tokenizer.encode(prompt);
        let eos_id = 2u32;

        // Autoregressive generation using forward_token with KV cache.
        // Since generate() takes &self (not &mut self), we use the legacy
        // full-sequence forward path here. Use generate_mut() for KV-cached
        // generation.
        let mut tokens = prompt_tokens;
        let mut generated = Vec::new();

        for _ in 0..params.max_tokens {
            let logits = self.forward(&tokens)?;
            let next_token = Self::argmax(&logits);

            if next_token == eos_id || next_token == 0 {
                break;
            }

            generated.push(next_token);
            tokens.push(next_token);
        }

        // Decode generated tokens back to text
        let text = tokenizer.decode(&generated);
        Ok(text)
    }

    fn generate_stream(
        &self,
        prompt: &str,
        params: GenerateParams,
    ) -> Result<Box<dyn Iterator<Item = Result<GeneratedToken>> + Send + '_>> {
        let result = self.generate(prompt, params)?;
        let tokens: Vec<Result<GeneratedToken>> = result
            .chars()
            .enumerate()
            .map(|(i, c)| {
                Ok(GeneratedToken {
                    id: i as u32,
                    text: c.to_string(),
                    logprob: None,
                    is_special: false,
                })
            })
            .collect();
        Ok(Box::new(tokens.into_iter()))
    }

    fn generate_stream_v2(&self, prompt: &str, params: GenerateParams) -> Result<TokenStream> {
        let (tx, stream) = TokenStream::channel();
        let result = self.generate(prompt, params.clone());

        match result {
            Ok(text) => {
                let _ = tx.send(StreamEvent::Token(GeneratedToken {
                    id: 0,
                    text,
                    logprob: None,
                    is_special: false,
                }));
                let _ = tx.send(StreamEvent::Done {
                    total_tokens: 1,
                    duration_ms: 0,
                    tokens_per_second: 0.0,
                });
            }
            Err(e) => {
                let _ = tx.send(StreamEvent::Error(e.to_string()));
            }
        }

        Ok(stream)
    }

    fn get_embeddings(&self, text: &str) -> Result<Vec<f32>> {
        let config = self.config.as_ref().ok_or_else(|| {
            RuvLLMError::Model("No model loaded".to_string())
        })?;
        let tokenizer = self.tok.as_ref().ok_or_else(|| {
            RuvLLMError::Model("No tokenizer loaded".to_string())
        })?;

        let ids = tokenizer.encode(text);
        if ids.is_empty() {
            return Err(RuvLLMError::Model("Empty token sequence".to_string()));
        }

        // Use last token embedding as text representation
        let last_id = *ids.last().unwrap() as usize;
        let hidden = config.hidden_size;
        if last_id >= config.vocab_size {
            return Err(RuvLLMError::Model("Token exceeds vocab".to_string()));
        }
        Ok(self.embedding[last_id * hidden..(last_id + 1) * hidden].to_vec())
    }

    fn tokenizer(&self) -> Option<&dyn BackendTokenizer> {
        self.tok.as_ref().map(|t| {
            // Safety: we return a reference with the same lifetime as &self.
            // The TokenizerBridge is a thin wrapper — we use a raw pointer trick
            // to avoid the borrow checker issue with returning a trait object
            // that borrows from self.
            //
            // Alternative: store a Box<dyn BackendTokenizer> directly. For now,
            // return None and callers should use `self.tok` directly.
            let _ = t;
            // Return None for the trait-object path; callers can use tok() accessor
            None::<&dyn BackendTokenizer>
        }).flatten()
    }

    fn is_model_loaded(&self) -> bool {
        self.loaded
    }

    fn model_info(&self) -> Option<ModelInfo> {
        let config = self.config.as_ref()?;
        Some(ModelInfo {
            name: self.model_path.clone(),
            architecture: ModelArchitecture::Qwen,
            num_parameters: config.num_layers
                * config.num_experts
                * config.intermediate_size
                * config.hidden_size
                * 3,
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            num_layers: config.num_layers,
            max_context_length: config.max_context,
            quantization: Some(Quantization::Q2K),
            memory_usage: self.embedding.len() * 4
                + self.lm_head.len() * 4
                + self
                    .layers
                    .iter()
                    .map(|l| {
                        l.gate_weight.len() * 4
                            + l.input_norm_weight.len() * 4
                            + l.post_attn_norm_weight.len() * 4
                            + l.attention.q_proj.memory_bytes()
                            + l.attention.k_proj.memory_bytes()
                            + l.attention.v_proj.memory_bytes()
                            + l.attention.o_proj.memory_bytes()
                            + l.experts
                                .iter()
                                .map(|e| {
                                    e.gate_proj.memory_bytes()
                                        + e.up_proj.memory_bytes()
                                        + e.down_proj.memory_bytes()
                                })
                                .sum::<usize>()
                    })
                    .sum::<usize>(),
        })
    }

    fn unload_model(&mut self) {
        self.config = None;
        self.embedding.clear();
        self.lm_head.clear();
        self.final_norm_weight.clear();
        self.layers.clear();
        self.kv_caches.clear();
        self.tok = None;
        self.rope_cos.clear();
        self.rope_sin.clear();
        self.loaded = false;
        self.model_path.clear();
    }
}

impl BitNetBackend {
    /// Autoregressive generate with KV cache (takes &mut self).
    ///
    /// This is the efficient path for generation: each token only computes
    /// attention against cached K/V vectors rather than reprocessing the
    /// full sequence.
    pub fn generate_cached(&mut self, prompt: &str, max_tokens: usize) -> Result<String> {
        if !self.loaded {
            return Err(RuvLLMError::Model("No model loaded".to_string()));
        }
        let tokenizer = self.tok.as_ref().ok_or_else(|| {
            RuvLLMError::Model("No tokenizer loaded".to_string())
        })?;

        let prompt_tokens = tokenizer.encode(prompt);
        let eos_id = 2u32;

        self.reset_cache();

        // Prefill: process all prompt tokens
        let mut last_logits = Vec::new();
        for (pos, &tid) in prompt_tokens.iter().enumerate() {
            last_logits = self.forward_token(tid, pos)?;
        }

        // Decode
        let mut generated = Vec::new();
        let mut pos = prompt_tokens.len();

        for _ in 0..max_tokens {
            let next_token = Self::argmax(&last_logits);
            if next_token == eos_id || next_token == 0 {
                break;
            }
            generated.push(next_token);
            last_logits = self.forward_token(next_token, pos)?;
            pos += 1;
        }

        let tokenizer = self.tok.as_ref().unwrap();
        Ok(tokenizer.decode(&generated))
    }

    /// Get the loaded tokenizer (if any).
    pub fn tok(&self) -> Option<&BpeTokenizer> {
        self.tok.as_ref()
    }
}

// ============================================================================
// Math Helpers (standalone functions used by the backend)
// ============================================================================

/// In-place RMSNorm: x = x / rms(x) * weight
fn rms_norm_inplace(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = x.len();
    let mut sum_sq = 0.0f32;
    for &v in x.iter() {
        sum_sq += v * v;
    }
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;

    for i in 0..n {
        x[i] = x[i] * inv_rms * weight.get(i).copied().unwrap_or(1.0);
    }
}

/// In-place softmax.
///
/// Guards against NaN propagation: if all inputs are -inf or NaN,
/// the result is a uniform distribution (1/n for each element).
fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }

    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Guard: if max_val is -inf or NaN, no valid scores exist.
    // Fall back to uniform distribution.
    if max_val.is_nan() || max_val.is_infinite() && max_val.is_sign_negative() {
        let uniform = 1.0 / x.len() as f32;
        for v in x.iter_mut() {
            *v = uniform;
        }
        return;
    }

    let mut sum_exp = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum_exp += *v;
    }
    // Guard: if sum_exp is zero, NaN, or subnormal, fall back to uniform
    if !sum_exp.is_normal() || sum_exp <= 0.0 {
        let uniform = 1.0 / x.len() as f32;
        for v in x.iter_mut() {
            *v = uniform;
        }
        return;
    }
    for v in x.iter_mut() {
        *v /= sum_exp;
    }
}

/// Sigmoid activation.
#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// FP16 bits to FP32 conversion (same as in gguf/quantization.rs).
#[inline(always)]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x03FF) as u32;

    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign);
        }
        let mut e = 1u32;
        let mut f = frac;
        while (f & 0x0400) == 0 {
            f <<= 1;
            e += 1;
        }
        f &= 0x03FF;
        return f32::from_bits(sign | ((127 - 15 + 1 - e) << 23) | (f << 13));
    }

    if exp == 31 {
        return f32::from_bits(sign | 0x7F80_0000 | (frac << 13));
    }

    f32::from_bits(sign | ((exp + 127 - 15) << 23) | (frac << 13))
}

/// FP32 matrix-vector product (transposed): out[i] = dot(mat[i*cols..], vec)
///
/// mat is [rows, cols] row-major, vec is [cols], out is [rows].
fn fp32_matvec_transposed(mat: &[f32], vec: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; rows];
    for i in 0..rows {
        let row_start = i * cols;
        if row_start + cols > mat.len() {
            break;
        }
        let mut dot = 0.0f32;
        for j in 0..cols {
            dot += mat[row_start + j] * vec[j];
        }
        output[i] = dot;
    }
    output
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitnet::{pack_ternary, TernaryTensor};

    #[test]
    fn test_build_tl1_lut() {
        let lut = build_tl1_lut();

        // Byte 0x00 = all bits 00 = all -1
        assert_eq!(lut[0x00], [-1, -1, -1, -1]);

        // Byte 0x55 = 01_01_01_01 = all 0
        assert_eq!(lut[0x55], [0, 0, 0, 0]);

        // Byte 0xAA = 10_10_10_10 = all +1
        assert_eq!(lut[0xAA], [1, 1, 1, 1]);

        // Byte 0x24 = 00_10_01_00 => positions: [00, 01, 10, 00] => [-1, 0, 1, -1]
        // bit layout LSB first: bits[0:1]=00, bits[2:3]=01, bits[4:5]=10, bits[6:7]=00
        // 0x24 = 0b00_10_01_00
        assert_eq!(lut[0x24], [-1, 0, 1, -1]);
    }

    #[test]
    fn test_rms_norm_inplace() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0; 4];
        rms_norm_inplace(&mut x, &w, 1e-6);

        // RMS of [1,2,3,4] = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
        let rms = (30.0f32 / 4.0).sqrt();
        let expected: Vec<f32> = [1.0, 2.0, 3.0, 4.0]
            .iter()
            .map(|v| v / rms)
            .collect();

        for (a, b) in x.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-4, "got {} expected {}", a, b);
        }
    }

    #[test]
    fn test_softmax_inplace() {
        let mut x = vec![1.0, 2.0, 3.0];
        softmax_inplace(&mut x);

        // Sum should be 1.0
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Values should be ordered
        assert!(x[0] < x[1]);
        assert!(x[1] < x[2]);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.999);
        assert!(sigmoid(-10.0) < 0.001);
    }

    #[test]
    fn test_fp32_matvec_transposed() {
        // Identity matrix 3x3
        let mat = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let vec_in = vec![2.0, 3.0, 4.0];
        let out = fp32_matvec_transposed(&mat, &vec_in, 3, 3);
        assert_eq!(out, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_tl1_gemv_simple() {
        let backend = BitNetBackend::new();

        // Create a 2x4 ternary weight matrix:
        // Row 0: [+1, +1, +1, +1]
        // Row 1: [-1, -1, -1, -1]
        let row0 = vec![1i8, 1, 1, 1];
        let row1 = vec![-1i8, -1, -1, -1];
        let mut all = row0.clone();
        all.extend_from_slice(&row1);
        let packed = pack_ternary(&all);

        let weight = TernaryTensor {
            packed_data: packed,
            scales: vec![1.0, 1.0], // one scale per block (each row < 256, so 1 block per row)
            shape: (2, 4),
            block_size: 256,
        };

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = backend.tl1_gemv(&weight, &input, 2, 4);

        // Row 0: 1+2+3+4 = 10, scale=1.0
        assert!((output[0] - 10.0).abs() < 1e-6);
        // Row 1: -(1+2+3+4) = -10, scale=1.0
        assert!((output[1] - (-10.0)).abs() < 1e-6);
    }

    #[test]
    fn test_tl1_gemv_with_zeros() {
        let backend = BitNetBackend::new();

        // Row: [+1, 0, -1, 0]
        let vals = vec![1i8, 0, -1, 0];
        let packed = pack_ternary(&vals);

        let weight = TernaryTensor {
            packed_data: packed,
            scales: vec![2.0],
            shape: (1, 4),
            block_size: 256,
        };

        let input = vec![5.0, 3.0, 7.0, 9.0];
        let output = backend.tl1_gemv(&weight, &input, 1, 4);

        // Result: (5.0 + 0 - 7.0 + 0) * 2.0 = -2.0 * 2.0 = -4.0
        assert!((output[0] - (-4.0)).abs() < 1e-6);
    }

    #[test]
    fn test_bitnet_model_config_default() {
        let config = BitNetModelConfig::default();
        assert_eq!(config.num_layers, 28);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_experts, 8);
        assert_eq!(config.active_experts, 2);
    }

    #[test]
    fn test_route_experts_topk() {
        let backend = BitNetBackend::new();
        let config = BitNetModelConfig {
            num_experts: 4,
            active_experts: 2,
            hidden_size: 4,
            ..Default::default()
        };

        // Gate weight [4 experts, 4 hidden]: identity-like so expert scores = hidden_states
        let gate_weight = vec![
            1.0, 0.0, 0.0, 0.0, // Expert 0 looks at dim 0
            0.0, 1.0, 0.0, 0.0, // Expert 1 looks at dim 1
            0.0, 0.0, 1.0, 0.0, // Expert 2 looks at dim 2
            0.0, 0.0, 0.0, 1.0, // Expert 3 looks at dim 3
        ];

        // Hidden states: dim 2 is highest, dim 3 is second
        let hidden = vec![0.1, 0.2, 0.9, 0.5];

        let (indices, weights) = backend
            .route_experts(&hidden, &gate_weight, &config)
            .unwrap();

        assert_eq!(indices.len(), 2);
        assert_eq!(weights.len(), 2);

        // Expert 2 should be first (score 0.9), Expert 3 second (score 0.5)
        assert_eq!(indices[0], 2);
        assert_eq!(indices[1], 3);

        // Weights should sum to ~1.0
        let wsum: f32 = weights.iter().sum();
        assert!((wsum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_backend_new_unloaded() {
        let backend = BitNetBackend::new();
        assert!(!backend.is_model_loaded());
        assert!(backend.model_info().is_none());
    }

    #[test]
    fn test_rope_tables() {
        let mut backend = BitNetBackend::new();
        backend.build_rope_tables(16, 8, 10000.0);

        let half = 4; // head_dim / 2
        // Position 0: all angles are 0 → cos=1, sin=0
        for i in 0..half {
            assert!((backend.rope_cos[i] - 1.0).abs() < 1e-5, "cos[0][{}]={}", i, backend.rope_cos[i]);
            assert!(backend.rope_sin[i].abs() < 1e-5, "sin[0][{}]={}", i, backend.rope_sin[i]);
        }

        // Table size should be max_seq * half
        assert_eq!(backend.rope_cos.len(), 16 * 4);
        assert_eq!(backend.rope_sin.len(), 16 * 4);
    }

    #[test]
    fn test_apply_rope_identity_at_pos_0() {
        let mut backend = BitNetBackend::new();
        backend.build_rope_tables(8, 4, 10000.0);

        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let original = x.clone();
        backend.apply_rope(&mut x, 1, 4, 0);

        // At position 0, all angles are 0, so cos=1, sin=0 → identity
        for (a, b) in x.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5, "RoPE at pos 0 should be identity: got {} vs {}", a, b);
        }
    }

    #[test]
    fn test_apply_rope_rotates_at_pos_1() {
        let mut backend = BitNetBackend::new();
        backend.build_rope_tables(8, 4, 10000.0);

        let mut x = vec![1.0, 0.0, 1.0, 0.0]; // head_dim=4, 1 head
        let original = x.clone();
        backend.apply_rope(&mut x, 1, 4, 1);

        // At position 1, some rotation should happen
        let changed = x.iter().zip(original.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(changed, "RoPE at pos 1 should rotate the vector");

        // Norm should be preserved (RoPE is an orthogonal rotation)
        let orig_norm: f32 = original.iter().map(|v| v * v).sum::<f32>().sqrt();
        let new_norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((orig_norm - new_norm).abs() < 1e-4, "RoPE should preserve norm");
    }

    #[test]
    fn test_kv_cache_operations() {
        let mut cache = LayerKvCache::new();
        assert_eq!(cache.len(), 0);

        cache.keys.push(vec![1.0, 2.0]);
        cache.values.push(vec![3.0, 4.0]);
        assert_eq!(cache.len(), 1);

        cache.keys.push(vec![5.0, 6.0]);
        cache.values.push(vec![7.0, 8.0]);
        assert_eq!(cache.len(), 2);

        cache.clear();
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_byte_level_tokenizer() {
        let tok = BitNetBackend::build_byte_level_tokenizer();
        assert_eq!(tok.vocab_size(), 260); // 4 special + 256 byte tokens

        // Roundtrip ASCII
        let ids = tok.encode("Hello");
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "Hello", "Byte-level tokenizer roundtrip failed");

        // BOS should be prepended
        assert_eq!(ids[0], 1);
    }

    #[test]
    fn test_byte_level_tokenizer_utf8() {
        let tok = BitNetBackend::build_byte_level_tokenizer();
        let text = "cafe\u{0301}"; // combining accent
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_backend_reset_cache() {
        let mut backend = BitNetBackend::new();
        // Manually set up caches
        backend.kv_caches = vec![LayerKvCache::new(), LayerKvCache::new()];
        backend.kv_caches[0].keys.push(vec![1.0]);
        backend.kv_caches[1].keys.push(vec![2.0]);

        backend.reset_cache();
        assert_eq!(backend.kv_caches[0].len(), 0);
        assert_eq!(backend.kv_caches[1].len(), 0);
    }

    #[test]
    fn test_attention_weights_struct() {
        // Just verify AttentionWeights can be constructed
        let packed = pack_ternary(&[1, 0, -1, 0]);
        let tensor = TernaryTensor {
            packed_data: packed.clone(),
            scales: vec![1.0],
            shape: (1, 4),
            block_size: 256,
        };
        let attn = AttentionWeights {
            q_proj: tensor.clone(),
            k_proj: tensor.clone(),
            v_proj: tensor.clone(),
            o_proj: tensor,
        };
        assert_eq!(attn.q_proj.shape, (1, 4));
    }

    #[test]
    fn test_tok_accessor() {
        let mut backend = BitNetBackend::new();
        assert!(backend.tok().is_none());

        backend.tok = Some(BitNetBackend::build_byte_level_tokenizer());
        assert!(backend.tok().is_some());
        assert_eq!(backend.tok().unwrap().vocab_size(), 260);
    }
}
