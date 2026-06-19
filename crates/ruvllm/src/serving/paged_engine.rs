//! Paged continuous-batching engine (ADR-258 Phase 6 engine swap-in).
//!
//! [`PagedBatchEngine`] drives a real continuous-batching loop over the paged
//! KV stack — [`PagedKvCacheManager`] + the block scheduler — rather than the
//! default engine's slot-based, simulated path. Each `step()`:
//!
//! 1. **Admits** waiting requests (prefill), sharing the longest cached prefix
//!    and respecting the block budget / watermark; queued requests are retried
//!    on later steps, and admission may preempt running requests under pressure.
//! 2. **Decodes** one token per running request, appending its KV to the cache
//!    (block growth + copy-on-write handled by the manager).
//! 3. **Retires** finished requests, freeing their blocks (shared prefix blocks
//!    survive for siblings).
//!
//! Token/KV production is abstracted behind [`TokenGenerator`] so the loop is
//! model-agnostic and unit-testable; a real build plugs in the model's decode
//! step (optionally using [`PagedKvCacheManager::attention`]). This module is
//! gated behind the `paged-kv` feature and does not touch the default engine.

use super::request::{FinishReason, RequestId, TokenOutput};
use super::paged_kv_manager::{PagedKvCacheManager, PagedKvManagerStats};
use crate::error::Result;
use crate::paged_kv::{PagedKvConfig, SchedulerConfig};
use std::collections::{HashMap, VecDeque};

/// Produces the next decode-step token and its KV for a request.
///
/// `seq_len` is the request's current context length (prompt + already-generated
/// tokens). Returns `(token_id, keys, values)` where `keys`/`values` are
/// `token_stride` elements each (one token's worth of KV).
pub trait TokenGenerator {
    /// Generate the next token + its KV for `req` at context length `seq_len`.
    fn next_token(&mut self, req: RequestId, seq_len: usize) -> (u32, Vec<f32>, Vec<f32>);
}

/// A request awaiting admission: prompt tokens + precomputed prompt KV.
pub struct PagedRequest {
    /// Request id.
    pub id: RequestId,
    /// Prompt token ids.
    pub prompt_tokens: Vec<u32>,
    /// Prompt keys (token-major, `prompt_tokens.len() * token_stride`).
    pub prompt_keys: Vec<f32>,
    /// Prompt values (same shape as keys).
    pub prompt_values: Vec<f32>,
    /// Maximum tokens to generate before finishing with `Length`.
    pub max_new_tokens: usize,
}

/// Internal per-running-request state.
struct RunningState {
    seq_len: usize,
    generated: usize,
    max_new_tokens: usize,
}

/// Engine configuration.
#[derive(Debug, Clone)]
pub struct PagedBatchEngineConfig {
    /// Paged cache configuration.
    pub cache: PagedKvConfig,
    /// Scheduler (admission/preemption) configuration.
    pub scheduler: SchedulerConfig,
}

impl Default for PagedBatchEngineConfig {
    fn default() -> Self {
        Self {
            cache: PagedKvConfig::default(),
            scheduler: SchedulerConfig::default(),
        }
    }
}

/// Outcome of a single engine step.
#[derive(Debug, Default, Clone)]
pub struct StepOutput {
    /// Requests admitted (prefilled) this step.
    pub admitted: Vec<RequestId>,
    /// Requests preempted back to the waiting queue this step.
    pub preempted: Vec<RequestId>,
    /// Tokens emitted by decode this step.
    pub tokens: Vec<TokenOutput>,
    /// Requests that finished and were freed this step.
    pub completed: Vec<RequestId>,
}

impl StepOutput {
    /// Whether the engine did any work this step.
    pub fn is_idle(&self) -> bool {
        self.admitted.is_empty()
            && self.preempted.is_empty()
            && self.tokens.is_empty()
            && self.completed.is_empty()
    }
}

/// Continuous-batching engine over the paged KV stack.
pub struct PagedBatchEngine {
    manager: PagedKvCacheManager,
    pending: VecDeque<PagedRequest>,
    running: HashMap<RequestId, RunningState>,
    /// Stable order of running ids so decode output is deterministic.
    running_order: Vec<RequestId>,
}

impl PagedBatchEngine {
    /// Create an engine with the given configuration.
    pub fn new(config: PagedBatchEngineConfig) -> Self {
        let manager = PagedKvCacheManager::new(config.cache, config.scheduler);
        Self {
            manager,
            pending: VecDeque::new(),
            running: HashMap::new(),
            running_order: Vec::new(),
        }
    }

    /// Enqueue a request for admission.
    pub fn submit(&mut self, req: PagedRequest) {
        self.pending.push_back(req);
    }

    /// Number of requests waiting for admission.
    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    /// Number of running requests.
    pub fn running_len(&self) -> usize {
        self.running.len()
    }

    /// Combined cache + scheduler telemetry.
    pub fn stats(&self) -> PagedKvManagerStats {
        self.manager.stats()
    }

    /// Shared access to the underlying manager (e.g. to run attention).
    pub fn manager(&self) -> &PagedKvCacheManager {
        &self.manager
    }

    /// Run one continuous-batching step.
    pub fn step<G: TokenGenerator>(&mut self, generator: &mut G) -> Result<StepOutput> {
        let mut out = StepOutput::default();

        // --- 1. Admission / prefill -----------------------------------------
        // Drain the waiting queue greedily; stop at the first request that does
        // not fit so FIFO order is preserved (head-of-line, like vLLM).
        let mut requeue: VecDeque<PagedRequest> = VecDeque::new();
        while let Some(req) = self.pending.pop_front() {
            let outcome = self.manager.admit_prefill(
                req.id,
                &req.prompt_tokens,
                &req.prompt_keys,
                &req.prompt_values,
            )?;
            // Record any preemptions. The scheduler already freed the victims'
            // blocks, so translate each SeqId back to its RequestId and drop the
            // running state *without* re-freeing (recompute policy: the caller
            // re-`submit`s them with their prompt KV later).
            for &victim_seq in &outcome.preempted {
                if let Some(req) = self.manager.forget_preempted_seq(victim_seq) {
                    self.running.remove(&req);
                    out.preempted.push(req);
                }
            }
            if outcome.admitted {
                let seq_len = req.prompt_tokens.len();
                self.running.insert(
                    req.id,
                    RunningState {
                        seq_len,
                        generated: 0,
                        max_new_tokens: req.max_new_tokens,
                    },
                );
                self.running_order.push(req.id);
                out.admitted.push(req.id);
            } else {
                // Did not fit; stop admitting and requeue the rest in order.
                requeue.push_back(req);
                break;
            }
        }
        // Anything left after a stop stays pending, order preserved.
        while let Some(req) = requeue.pop_front() {
            self.pending.push_front(req);
        }

        // --- 2. Decode ------------------------------------------------------
        let order = self.running_order.clone();
        for id in order {
            let seq_len = match self.running.get(&id) {
                Some(s) => s.seq_len,
                None => continue, // was preempted/completed this step
            };
            let (token, keys, values) = generator.next_token(id, seq_len);
            // Budget-checked single-token append; on OOM, preempt this request.
            match self.manager.extend(id, &[token], &keys, &values) {
                Ok(()) => {}
                Err(_) => {
                    self.handle_preemption(id, &mut out);
                    continue;
                }
            }
            let state = self.running.get_mut(&id).expect("running");
            state.seq_len += 1;
            state.generated += 1;
            let is_final = state.generated >= state.max_new_tokens;
            out.tokens.push(TokenOutput {
                request_id: id,
                token_id: token,
                token_text: None,
                logprob: None,
                is_final,
                finish_reason: is_final.then_some(FinishReason::Length),
                seq_len: state.seq_len,
            });
            if is_final {
                self.manager.free(id)?;
                self.running.remove(&id);
                out.completed.push(id);
            }
        }
        // Keep running_order in sync (drop completed/preempted ids).
        self.running_order.retain(|id| self.running.contains_key(id));

        Ok(out)
    }

    /// Drive the engine until all submitted requests complete, returning every
    /// emitted token in order. Bounded by `max_steps` to guarantee termination.
    pub fn run_to_completion<G: TokenGenerator>(
        &mut self,
        generator: &mut G,
        max_steps: usize,
    ) -> Result<Vec<TokenOutput>> {
        let mut all = Vec::new();
        for _ in 0..max_steps {
            if self.pending.is_empty() && self.running.is_empty() {
                break;
            }
            let out = self.step(generator)?;
            all.extend(out.tokens);
        }
        Ok(all)
    }

    /// Free a preempted request and record it. The engine drops its running
    /// state; under a recompute policy the caller re-`submit`s it later.
    fn handle_preemption(&mut self, id: RequestId, out: &mut StepOutput) {
        if self.running.remove(&id).is_some() {
            // Scheduler already freed the blocks via its preemption path; ensure
            // the manager's id mapping is cleared too.
            let _ = self.manager.free(id);
            out.preempted.push(id);
        }
    }
}

#[cfg(test)]
mod paged_engine_tests {
    use super::*;

    fn cfg(total_blocks: usize) -> PagedBatchEngineConfig {
        PagedBatchEngineConfig {
            cache: PagedKvConfig {
                block_size: 4,
                num_kv_heads: 1,
                head_dim: 2,
                total_blocks,
                verify_prefix_tokens: false,
            },
            scheduler: SchedulerConfig {
                max_running: 64,
                watermark_blocks: 1,
                allow_preemption: true,
            },
        }
    }

    /// Generator that emits a fixed token and one token's worth of constant KV.
    struct ConstGen {
        stride: usize,
    }
    impl TokenGenerator for ConstGen {
        fn next_token(&mut self, _req: RequestId, seq_len: usize) -> (u32, Vec<f32>, Vec<f32>) {
            // Token id encodes position so outputs are checkable.
            (seq_len as u32, vec![0.5; self.stride], vec![0.25; self.stride])
        }
    }

    fn mk_request(id: RequestId, tokens: Vec<u32>, stride: usize, max_new: usize) -> PagedRequest {
        let n = tokens.len();
        PagedRequest {
            id,
            prompt_tokens: tokens,
            prompt_keys: vec![0.5; n * stride],
            prompt_values: vec![0.25; n * stride],
            max_new_tokens: max_new,
        }
    }

    #[test]
    fn single_request_prefill_decode_complete() {
        let config = cfg(64);
        let stride = config.cache.token_stride();
        let mut engine = PagedBatchEngine::new(config);
        let req = RequestId::new();
        engine.submit(mk_request(req, (0..6).collect(), stride, 3));

        let mut gen = ConstGen { stride };
        let tokens = engine.run_to_completion(&mut gen, 100).unwrap();
        // 3 generated tokens, last is final.
        assert_eq!(tokens.len(), 3);
        assert!(tokens.last().unwrap().is_final);
        assert_eq!(tokens.last().unwrap().finish_reason, Some(FinishReason::Length));
        // Everything freed.
        assert_eq!(engine.running_len(), 0);
        assert_eq!(engine.stats().cache.pool.allocated_blocks, 0);
    }

    #[test]
    fn many_requests_share_prefix_under_budget() {
        let config = cfg(256);
        let stride = config.cache.token_stride();
        let mut engine = PagedBatchEngine::new(config);

        // 16-token shared prefix (4 blocks) + unique 4-token suffix each.
        let shared: Vec<u32> = (7000..7016).collect();
        let n = 8;
        for i in 0..n {
            let mut toks = shared.clone();
            toks.extend_from_slice(&[8000 + i, 8001 + i, 8002 + i, 8003 + i]);
            engine.submit(mk_request(RequestId::new(), toks, stride, 2));
        }

        let mut gen = ConstGen { stride };
        // One step admits everyone (budget permitting) and decodes once.
        let s1 = engine.step(&mut gen).unwrap();
        assert_eq!(s1.admitted.len(), n as usize);
        // Shared prefix amortized: 4 shared + 8 unique suffix = 12 prompt blocks
        // (decode tokens may have added at most one more per seq).
        let alloc = engine.stats().cache.pool.allocated_blocks;
        assert!(alloc <= 12 + n as usize, "alloc {alloc}");
        assert!(engine.stats().cache.prefix_hit_tokens >= 16 * (n as u64 - 1));

        // Finish the rest.
        let all = engine.run_to_completion(&mut gen, 100).unwrap();
        assert!(!all.is_empty());
        assert_eq!(engine.running_len(), 0);
        assert_eq!(engine.stats().cache.pool.allocated_blocks, 0);
    }

    #[test]
    fn admission_backpressure_queues_when_full() {
        // Tiny pool: 4 blocks, watermark 1. Each 8-token prompt needs 2 blocks.
        let config = cfg(4);
        let stride = config.cache.token_stride();
        let mut engine = PagedBatchEngine::new(PagedBatchEngineConfig {
            scheduler: SchedulerConfig {
                allow_preemption: false,
                watermark_blocks: 1,
                max_running: 64,
            },
            ..config
        });
        // Distinct prompts so no sharing masks pressure.
        engine.submit(mk_request(RequestId::new(), (10..18).collect(), stride, 1));
        engine.submit(mk_request(RequestId::new(), (100..108).collect(), stride, 1));

        let mut gen = ConstGen { stride };
        let s1 = engine.step(&mut gen).unwrap();
        // First fits (2 blocks, free 4 >= 2+1); second needs 2 but free now 2 <
        // 2+1 -> queued.
        assert_eq!(s1.admitted.len(), 1);
        assert_eq!(engine.pending_len(), 1);
    }
}
