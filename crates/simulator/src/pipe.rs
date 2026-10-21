// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU32, Ordering},
};

use gem_cuda::ucci;
use indexmap::IndexMap;
#[cfg(not(feature = "unified-memory"))]
use ulib::PinnedMappedBuffer;
#[cfg(feature = "unified-memory")]
use ulib::UnifiedBuffer;
use ulib::{AsUPtrMut, Device, RawUPtrMut, UVec};

/// Pipe buffer type: `PinnedMappedBuffer` by default (PCIe-coherent zero-copy),
/// or `UnifiedBuffer` when the `unified-memory` feature is enabled
/// (NVLink-C2C).
#[cfg(feature = "unified-memory")]
type DpiPipeBuf = UnifiedBuffer<u32>;
#[cfg(not(feature = "unified-memory"))]
type DpiPipeBuf = PinnedMappedBuffer<u32>;

/// CPU-side slot info for the poll thread.
#[derive(Clone)]
pub(crate) struct DpiPipeSlotInfo {
    pub pipe_slot_start: usize,
    pub n_in_words: usize,
    pub n_out_words: usize,
    pub func_name: String,
    /// One bit per input bit; set if the GPU-gathered bit must be inverted
    /// before passing to the callback (handles synthesis-introduced
    /// inversions).
    pub in_invert_mask: Vec<u32>,
    /// One bit per input bit; set if this input is a compile-time constant
    /// (the GPU-gathered value is meaningless and must be overridden).
    pub in_const_mask: Vec<u32>,
    /// For constant bits (where `in_const_mask` is set), the constant value.
    pub in_const_ones: Vec<u32>,
}

pub(crate) struct GpuPipeTables {
    pub in_offsets: Vec<u32>,
    /// Bit width of each input port, in port order across all calls.
    pub in_port_widths: Vec<u32>,
    pub out_offsets: Vec<u32>,
    /// Bit width of each output port, in port order across all calls.
    pub out_port_widths: Vec<u32>,
    pub call_descs: Vec<ucci::DpiCallDesc>,
    pub slot_infos: Vec<DpiPipeSlotInfo>,
    pub pipe_total_words: usize,
    /// Every-cycle call indices that could not be pipelined (And inputs).
    /// Must be executed on the CPU before each kernel launch.
    pub cpu_only_indices: Vec<usize>,
}

/// All GPU-side DPI pipe state for one simulator instance.
///
/// Holds the unified pipe buffer (CPU+GPU accessible via NVLink-C2C), the
/// flattened bit-offset tables, the per-call descriptors uploaded to the
/// device, and the CPU-side slot metadata for the poll thread.
pub(crate) struct GpuDpiPipe {
    /// Shared pipe buffer; `None` when there are no every-cycle DPI calls.
    pub pipe: Option<DpiPipeBuf>,
    pub in_offsets: UVec<u32>,
    pub in_port_widths: UVec<u32>,
    pub out_offsets: UVec<u32>,
    pub out_port_widths: UVec<u32>,
    pub call_descs: UVec<ucci::DpiCallDesc>,
    pub slot_infos: Vec<DpiPipeSlotInfo>,
    /// Every-cycle call indices that could not be pipelined. Executed on the
    /// CPU before each kernel launch via `DpiState::execute_by_indices`.
    pub cpu_only_indices: Vec<usize>,
}

impl GpuDpiPipe {
    /// Empty pipe for when there are no every-cycle DPI calls.
    pub fn empty() -> Self {
        Self {
            pipe: None,
            in_offsets: UVec::new(),
            in_port_widths: UVec::new(),
            out_offsets: UVec::new(),
            out_port_widths: UVec::new(),
            call_descs: UVec::new(),
            slot_infos: Vec::new(),
            cpu_only_indices: Vec::new(),
        }
    }

    /// Allocate and upload all GPU pipe state from pre-built tables.
    pub fn from_tables(
        tables: GpuPipeTables,
        device: Device,
    ) -> anyhow::Result<Self> {
        let dev_id = match device {
            Device::CUDA(dev_id) => dev_id,
            _ => return Ok(GpuDpiPipe::empty()),
        };

        let pipe = {
            let pipe_len = std::num::NonZeroUsize::new(tables.pipe_total_words)
                .ok_or_else(|| {
                    anyhow::anyhow!("DPI pipe would be zero-length")
                })?;

            #[cfg(feature = "unified-memory")]
            let mut pipe = UnifiedBuffer::<u32>::new(pipe_len, dev_id)
                .map_err(|e| anyhow::anyhow!("DPI pipe alloc: {:?}", e))?;
            #[cfg(not(feature = "unified-memory"))]
            let mut pipe = PinnedMappedBuffer::<u32>::new(pipe_len, dev_id)
                .map_err(|e| anyhow::anyhow!("DPI pipe alloc: {:?}", e))?;

            // Zero all status words via the CPU host pointer so the GPU kernel
            // doesn't see stale state on first launch.
            // SAFETY: pipe is properly aligned and a valid pointer
            unsafe {
                std::ptr::write_bytes(
                    pipe.as_mut_uptr(Device::CPU),
                    0,
                    tables.pipe_total_words,
                );
            }
            pipe
        };

        let mut in_offsets = UVec::from(tables.in_offsets);
        let mut in_port_widths = UVec::from(tables.in_port_widths);
        let mut out_offsets = UVec::from(tables.out_offsets);
        let mut out_port_widths = UVec::from(tables.out_port_widths);
        let mut call_descs = UVec::from(tables.call_descs);
        in_offsets.as_mut_uptr(device);
        in_port_widths.as_mut_uptr(device);
        out_offsets.as_mut_uptr(device);
        out_port_widths.as_mut_uptr(device);
        call_descs.as_mut_uptr(device);

        Ok(Self {
            pipe: Some(pipe),
            in_offsets,
            in_port_widths,
            out_offsets,
            out_port_widths,
            call_descs,
            slot_infos: tables.slot_infos,
            cpu_only_indices: tables.cpu_only_indices,
        })
    }

    pub fn has_cpu_only_calls(&self) -> bool {
        !self.cpu_only_indices.is_empty()
    }

    pub fn n_calls(&self) -> usize {
        self.call_descs.len()
    }

    pub fn open_pipe_channel(
        &mut self,
        device: Device,
        callbacks: IndexMap<
            String,
            Arc<dyn Fn(&[u32]) -> Vec<u32> + Send + Sync>,
        >,
    ) -> DpiPipeChannel {
        DpiPipeChannel::open(
            &mut self.pipe,
            device,
            self.slot_infos.clone(),
            callbacks,
        )
    }
}

/// Newtype wrapper that makes a `*mut u32` safe to send across thread
/// boundaries. Used only to pass the unified pipe buffer pointer to the poll
/// thread; the caller ensures the pointed-to memory outlives the thread.
struct SendablePipePtr(*mut u32);
// SAFETY: SendablePipePtr is only used to pass the pipe buffer pointer to a
// single poll thread; the caller guarantees the pointed-to memory outlives the
// thread and no other thread aliases it.
unsafe impl Send for SendablePipePtr {}
impl SendablePipePtr {
    fn get(&self) -> *mut u32 {
        self.0
    }
}

/// A channel connecting the GPU DPI pipe to a CPU poll thread.
///
/// `open` spawns the CPU poll thread and returns the channel. Pass
/// `pipe_buf` directly to the CUDA kernel, then call `shutdown()` once the
/// kernel returns.
///
/// When there are no pipelined DPI calls, `open` returns a channel with a
/// null `pipe_buf` and no thread; `shutdown` is then a no-op.
pub(crate) struct DpiPipeChannel {
    /// Ready-to-use kernel argument for the pipe buffer.
    pub pipe_buf: RawUPtrMut<u32>,
    shutdown: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl DpiPipeChannel {
    /// Open the channel. Always succeeds; spawns a poll thread only when
    /// there are pipelined DPI calls.
    pub fn open(
        pipe: &mut Option<DpiPipeBuf>,
        device: Device,
        slot_infos: Vec<DpiPipeSlotInfo>,
        callbacks: IndexMap<
            String,
            Arc<dyn Fn(&[u32]) -> Vec<u32> + Send + Sync>,
        >,
    ) -> Self {
        // Split into separate CPU host pointer (for the poll thread) and GPU
        // device pointer (for the kernel).
        let (cpu_ptr, gpu_ptr) = pipe
            .as_mut()
            .map(|p| (p.as_mut_uptr(Device::CPU), p.as_mut_uptr(device)))
            .unwrap_or((std::ptr::null_mut(), std::ptr::null_mut()));

        // SAFETY: kernel guards all dpi_pipe accesses with n_dpi_calls > 0,
        // so the null case is never dereferenced.
        let pipe_buf = unsafe { RawUPtrMut::new(gpu_ptr, device) };

        let shutdown = Arc::new(AtomicBool::new(false));
        let thread = if slot_infos.is_empty() {
            // tracing::debug!("[dpi-pipe] no slots, poll thread not started");
            None
        } else {
            let pipe = SendablePipePtr(cpu_ptr);
            let shutdown_clone = Arc::clone(&shutdown);
            Some(std::thread::spawn(move || {
                let pipe = pipe.get();
                loop {
                    let mut all_idle = true;
                    for slot in &slot_infos {
                        // SAFETY:
                        //
                        // - slot layout is `[status(1)][in_words][out_words]`
                        //   so `pipe_slot_start` is within the allocation
                        // - buffer guarantees 4-byte alignment
                        // - `AtomicU32` is `#[repr(transparent)]` over
                        //   `UnsafeCell<u32>` so the cast is layout-compatible
                        // - the ordering guarantee comes from the `Ordering`
                        //   argument to load/store, not from the cast - the
                        //   status word is exclusively accessed through atomic
                        //   operations on both CPU and GPU sides, so there are
                        //   no mixed atomic/non-atomic accesses to this word
                        let status = unsafe {
                            &*(pipe.add(slot.pipe_slot_start)
                                as *const AtomicU32)
                        };

                        // Acquire pairs with the GPU's `__threadfence_system` +
                        // `atomicExch` that posts status=1, ensuring the input
                        // words written before the fence are visible here.
                        if status.load(Ordering::Acquire) != 1 {
                            continue;
                        }
                        all_idle = false;

                        // NOTE `read_volatile` is required because the GPU
                        // wrote these words before updating the status to 2.
                        // Without volatile the compiler may cache a stale read.
                        // The loop is necessary because `read_volatile` is the
                        // only stable volatile write at the moment. Eventually
                        // we can use `volatile_copy_nonoverlapping_memory` once
                        // it hits stable.
                        let mut inputs: Vec<u32> = (0..slot.n_in_words)
                            .map(|i| {
                                // SAFETY:
                                //
                                // - pipe is valid for pipe_total_words u32s
                                // - construction guarantees
                                //   slot.pipe_slot_start + 1 + i <
                                //   pipe_total_words for i < n_in_words.
                                // - pointer is 4-byte aligned
                                unsafe {
                                    pipe.add(slot.pipe_slot_start + 1 + i)
                                        .read_volatile()
                                }
                            })
                            .collect();

                        #[allow(clippy::needless_range_loop)]
                        for i in 0..slot.n_in_words {
                            // apply inversions
                            inputs[i] ^= slot.in_invert_mask[i];
                            // and override constants (ignore GPU values)
                            inputs[i] = (inputs[i] & !slot.in_const_mask[i])
                                | slot.in_const_ones[i];
                        }

                        let outputs =
                            callbacks.get(&slot.func_name).map_or_else(
                                || vec![0u32; slot.n_out_words],
                                |cb| cb(&inputs),
                            );
                        assert_eq!(
                            outputs.len(),
                            slot.n_out_words,
                            "DPI callback '{}' returned {} words, expected {}",
                            slot.func_name,
                            outputs.len(),
                            slot.n_out_words,
                        );

                        // NOTE write_volatile is required because the GPU reads
                        // these words after we set status to 2. Without
                        // volatile the compiler may treat the writes as dead
                        // since no CPU code reads them back. The loop is
                        // necessary because `write_volatile` is the only stable
                        // volatile write at the moment. Eventually we can use
                        // `volatile_copy_nonoverlapping_memory` once it hits
                        // stable.
                        for i in 0..slot.n_out_words {
                            // SAFETY:
                            //
                            // - pipe is valid for pipe_total_words u32s
                            // - construction guarantees slot.pipe_slot_start +
                            //   1 + n_in_words + i < pipe_total_words for i <
                            // n_out_words
                            // - pointer is 4-byte aligned
                            //
                            unsafe {
                                pipe.add(
                                    slot.pipe_slot_start
                                        + 1
                                        + slot.n_in_words
                                        + i,
                                )
                                .write_volatile(
                                    outputs.get(i).copied().unwrap_or(0),
                                )
                            };
                        }

                        // This release pairs with the GPU's
                        // `__threadfence_system` after it reads status=2,
                        // ensuring the output words are visible before the GPU
                        // proceeds.
                        status.store(2, Ordering::Release);
                    }

                    if all_idle {
                        if shutdown_clone.load(Ordering::Relaxed) {
                            break;
                        }
                        // All slots are idle: the GPU is computing and hasn't
                        // posted any DPI requests yet.
                        // spin_loop() emits a PAUSE/YIELD hint: reduces power
                        // and avoids memory ordering pipeline stalls from the
                        // tight Acquire loads without adding latency to the
                        // response path.
                        std::hint::spin_loop();
                    }
                }
            }))
        };

        Self {
            pipe_buf,
            shutdown,
            thread,
        }
    }

    /// Signal the thread to stop and wait for it to exit.
    ///
    /// Call this after the CUDA kernel returns. By that point the GPU has reset
    /// all status words to 0, so the thread will find every slot idle and exit
    /// on the next iteration.
    pub fn shutdown(mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            thread.join().expect("DPI pipe channel thread panicked");
        }
    }
}
