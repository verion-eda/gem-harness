// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
//! CUDA kernel backend: wraps the GPU-side simulator.
//!
//! This crate is the only place CUDA source lives. Its build script
//! invokes [`ucc`] to compile `csrc/kernel_v1_impl.cuh` with `nvcc` and
//! auto-generate FFI bindings; consumers see a safe-ish Rust surface in
//! [`ucci`].
//!
//! Upstream: the `flatten` crate produces the `FlattenedScriptV1` that
//! ends up packed into the descriptors here. Downstream: linked into
//! [`simulator`] (under the `cuda` feature) and [`gem-simulator`].
//!
//! [`ucc`]: ../ucc/
//! [`flatten`]: ../flatten/
//! [`simulator`]: ../simulator/
//! [`gem-simulator`]: ../gem_simulator/

pub mod ucci {
    #![allow(clippy::undocumented_unsafe_blocks)]
    /// GPU-side descriptor for one DPI pipe slot.
    /// Layout must match `DpiCallDesc` in kernel_v1_impl_interactive.cuh (8 ×
    /// u32).
    #[repr(C)]
    #[derive(Clone, ulib::UniversalCopy)]
    pub struct DpiCallDesc {
        pub in_offsets_start: u32,
        pub n_in_words: u32,
        pub n_in_ports: u32,
        pub in_ports_start: u32,
        pub out_offsets_start: u32,
        pub n_out_ports: u32,
        pub out_ports_start: u32,
        pub pipe_slot_start: u32,
    }

    /// A stop watch - stops simulation when a signal matches an expected value.
    /// Kernel format: 3 u32s [offset, mask, expected]
    #[derive(Clone, Copy, Debug)]
    pub struct StopWatch {
        /// Word offset in state array
        pub state_offset: u32,
        /// Bit mask for the signal
        pub mask: u32,
        /// Expected value (after masking)
        pub expected: u32,
    }

    impl StopWatch {
        pub fn new(state_offset: u32, mask: u32, expected: u32) -> Self {
            Self {
                state_offset,
                mask,
                expected,
            }
        }

        /// Serialize to flat u32 array (8 elements)
        pub fn to_u32_array(&self) -> [u32; 3] {
            [self.state_offset, self.mask, self.expected]
        }
    }

    /// Clock edge info - describes a clock signal and its edge flag positions.
    /// Used by both CPU and CUDA simulators to detect edges by comparing
    /// input_state (new values) vs output_state (old values from previous
    /// cycle). Kernel format: 6 u32s [clock_word, clock_mask, pe_word,
    /// pe_mask, ne_word, ne_mask]
    #[derive(Clone, Copy, Debug)]
    pub struct ClockEdgeInfo {
        /// Word offset in state array for the clock signal
        pub clock_word: u32,
        /// Bit mask for the clock signal
        pub clock_mask: u32,
        /// Word offset for posedge flag (u32::MAX if none)
        pub posedge_word: u32,
        /// Bit mask for posedge flag
        pub posedge_mask: u32,
        /// Word offset for negedge flag (u32::MAX if none)
        pub negedge_word: u32,
        /// Bit mask for negedge flag
        pub negedge_mask: u32,
    }

    impl ClockEdgeInfo {
        pub fn to_u32_array(&self) -> [u32; 6] {
            [
                self.clock_word,
                self.clock_mask,
                self.posedge_word,
                self.posedge_mask,
                self.negedge_word,
                self.negedge_mask,
            ]
        }
    }

    /// A toggle watch - toggles a clock signal at a specified frequency.
    /// Includes edge flag positions for posedge/negedge detection.
    /// Kernel format: 8 u32s [offset, mask, start_offset, freq, pe_word,
    /// pe_mask, ne_word, ne_mask]
    #[derive(Clone, Copy, Debug)]
    pub struct ToggleWatch {
        pub edge: ClockEdgeInfo,
        /// Cycle offset to start toggling (0 = first cycle)
        pub start_offset: u32,
        /// Toggle frequency (1 = every cycle, 2 = every other cycle, etc.)
        pub freq: u32,
    }

    impl ToggleWatch {
        pub fn new(edge: ClockEdgeInfo, freq: u32, start_offset: u32) -> Self {
            Self {
                edge,
                start_offset,
                freq,
            }
        }

        /// Serialize to flat u32 array (8 elements)
        pub fn to_u32_array(&self) -> [u32; 8] {
            [
                self.edge.clock_word,
                self.edge.clock_mask,
                self.start_offset,
                self.freq,
                self.edge.posedge_word,
                self.edge.posedge_mask,
                self.edge.negedge_word,
                self.edge.negedge_mask,
            ]
        }
    }

    /// Watch condition - either stops simulation or toggles a signal
    #[derive(Clone, Copy, Debug)]
    pub enum WatchCondition {
        Stop(StopWatch),
        Toggle(ToggleWatch),
    }

    impl WatchCondition {
        /// Create a stop watch
        pub fn new_stop(state_offset: u32, mask: u32, expected: u32) -> Self {
            Self::Stop(StopWatch::new(state_offset, mask, expected))
        }

        /// Create a toggle watch with edge detection
        pub fn new_toggle(
            edge: ClockEdgeInfo,
            freq: u32,
            start_offset: u32,
        ) -> Self {
            Self::Toggle(ToggleWatch::new(edge, freq, start_offset))
        }
    }

    #[cfg(feature = "cuda")]
    include!(concat!(env!("OUT_DIR"), "/uccbind/kernel_v1.rs"));
}
