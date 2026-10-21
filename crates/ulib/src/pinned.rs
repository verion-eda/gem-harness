// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
//! Pinned Buffers
//!
//! All the buffers here should not be used when CPU and GPU perform
//! **concurrent** writes.
//!
//! Usage scenarios:
//! - PCI-e interconnect:
//!   - less than 1 MB: use zero copy mapped: [`PinnedMappedBuffer`]
//!   - more than 1 MB (bulk): explicit DMA copy -> [`PinnedStagingBuffer`] with
//!     `cuMemcopyAsync`
//! - NVLink-c2c interconnect: use [`UnifiedBuffer`]

use std::num::NonZeroUsize;

use crate::{AsUPtr, AsUPtrMut, Device, UniversalCopy};

/// A CUDA unified (managed) memory buffer accessible from both CPU and any
/// CUDA device via the same pointer.
///
/// Unlike [`PinnedMappedBuffer`], there is no separate host/device pointer —
/// the same virtual address is valid everywhere, so [`AsUPtr`]/[`AsUPtrMut`]
/// return the same pointer regardless of device.
///
/// # When to use
///
/// Allocated with `cuMemAllocManaged(CU_MEM_ATTACH_GLOBAL)`. The CUDA driver
/// migrates pages between host and device memory automatically on access,
/// making this suitable for NVLink-C2C and other architectures where the
/// migration cost is negligible or the access pattern is mixed.
///
/// # Note
///
/// Memory is not initialized on allocation; callers must write before reading.
pub struct UnifiedBuffer<T: UniversalCopy> {
    ptr: *mut T,
    len: NonZeroUsize,
    cuda_device: u8,
}

// SAFETY: `UnifiedBuffer` owns its allocation; the raw pointer `ptr` is not
// shared across threads without synchronisation.
unsafe impl<T: UniversalCopy> Send for UnifiedBuffer<T> {}
// SAFETY: All shared access goes through the CUDA-managed unified memory
// coherence protocol; no unsynchronised mutation of `ptr` is possible.
unsafe impl<T: UniversalCopy> Sync for UnifiedBuffer<T> {}

impl<T: UniversalCopy> UnifiedBuffer<T> {
    /// Allocate `len` elements of unified memory associated with `cuda_device`.
    ///
    /// Memory is uninitialized. Returns an error if the CUDA allocation fails.
    pub fn new(
        len: NonZeroUsize,
        cuda_device: u8,
    ) -> cust::error::CudaResult<Self> {
        use cust_raw::{
            CUmemAttach_flags_enum::CU_MEM_ATTACH_GLOBAL, cuMemAllocManaged,
            cudaError_enum::CUDA_SUCCESS,
        };

        const {
            assert!(
                std::mem::size_of::<T>() > 0,
                "T must not be a zero-sized type"
            );
        }

        let size_of_t = NonZeroUsize::new(std::mem::size_of::<T>()).unwrap();
        let byte_size = len
            .checked_mul(size_of_t)
            .ok_or(cust::error::CudaError::InvalidMemoryAllocation)?;

        let _context = Device::CUDA(cuda_device).get_context();

        let mut raw: cust_raw::CUdeviceptr = 0;
        // SAFETY: `raw` is a valid out-pointer, `byte_size` is non-zero (T is
        // non-ZST and len is NonZeroUsize), and the CUDA context is active.
        let result = unsafe {
            cuMemAllocManaged(
                &mut raw,
                byte_size.into(),
                CU_MEM_ATTACH_GLOBAL as u32,
            )
        };
        if result != CUDA_SUCCESS {
            return Err(cust::error::CudaError::UnknownError);
        }

        Ok(UnifiedBuffer {
            ptr: raw as *mut T,
            len,
            cuda_device,
        })
    }

    pub fn len(&self) -> NonZeroUsize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        false
    }
}

impl<T: UniversalCopy> Drop for UnifiedBuffer<T> {
    fn drop(&mut self) {
        let _context = Device::CUDA(self.cuda_device).get_context();
        // SAFETY: `self.ptr` was returned by `cuMemAllocManaged` in `new` and
        // has not been freed; the context is re-entered above.
        unsafe {
            cust_raw::cuMemFree_v2(self.ptr as cust_raw::CUdeviceptr);
        }
    }
}

impl<T: UniversalCopy> AsUPtr<T> for UnifiedBuffer<T> {
    fn as_uptr(&self, _device: Device) -> *const T {
        self.ptr as *const T
    }
}

impl<T: UniversalCopy> AsUPtrMut<T> for UnifiedBuffer<T> {
    fn as_mut_uptr(&mut self, _device: Device) -> *mut T {
        self.ptr
    }
}

/// A page-locked host staging buffer paired with a device buffer for
/// DMA-engine transfers via a CUDA stream.
///
/// Allocated with `cuMemAllocHost` (no mapping flags) on the host side and a
/// standard `DeviceBuffer<T>` on the device side.
///
/// # When to use
///
/// Use this when the GPU will read or write data multiple times: pay the PCIe
/// transfer cost once with [`copy_to_device`](Self::copy_to_device), then let
/// the SMs read repeatedly from fast device DRAM. This pattern is usually
/// refered to as "use pinned memory for transfers".
///
/// Because the host buffer is page-locked, the GPU's dedicated DMA copy engine
/// can operate on it directly - no bounce buffer, no OS page faults. The
/// transfer runs on the copy engine in parallel with SM computation on another
/// stream, achieving close to peak PCIe bandwidth. The SMs never touch the
/// host memory at all.
///
/// # Contrast with [`PinnedMappedBuffer`]
///
/// [`PinnedMappedBuffer`] skips the explicit copy step: the SMs read host DRAM
/// directly over PCIe on each access (`DEVICEMAP`). That is faster when data
/// is read at most once or twice, but slower for repeated reads because every
/// SM access becomes a PCIe round-trip with fine-grained latency rather than a
/// single bulk DMA transfer.
///
/// # Note
///
/// Memory is not initialized on allocation; callers must write before reading.
/// A future improvement would be to use `MaybeUninit<T>` as the element type
/// so the type system enforces initialization before reads.
pub struct PinnedStagingBuffer<T: UniversalCopy> {
    host_ptr: *mut T,
    device_buf: cust::memory::DeviceBuffer<T>,
    len: NonZeroUsize,
    cuda_device: u8,
}

// SAFETY: `PinnedStagingBuffer` owns its pinned host and device allocations;
// no unsynchronised aliasing exists across thread boundaries.
unsafe impl<T: UniversalCopy> Send for PinnedStagingBuffer<T> {}
// SAFETY: Shared access is safe because the host and device buffers are only
// mutated when the caller has exclusive access (enforced by `&mut self`).
unsafe impl<T: UniversalCopy> Sync for PinnedStagingBuffer<T> {}

impl<T: UniversalCopy> PinnedStagingBuffer<T> {
    /// Allocate `len` elements of pinned host memory and a matching device
    /// buffer on `cuda_device`. Memory is uninitialized.
    pub fn new(
        len: NonZeroUsize,
        cuda_device: u8,
    ) -> cust::error::CudaResult<Self> {
        use std::os::raw::c_void;

        use cust_raw::{
            cuMemAllocHost_v2, cuMemFreeHost, cudaError_enum::CUDA_SUCCESS,
        };

        const {
            assert!(
                std::mem::size_of::<T>() > 0,
                "T must not be a zero-sized type"
            );
        }

        let size_of_t = NonZeroUsize::new(std::mem::size_of::<T>()).unwrap();
        let byte_size = len
            .checked_mul(size_of_t)
            .ok_or(cust::error::CudaError::InvalidMemoryAllocation)?;

        let _context = Device::CUDA(cuda_device).get_context();

        let mut host_ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: `host_ptr` is a valid out-pointer and `byte_size` is
        // non-zero.
        let result =
            unsafe { cuMemAllocHost_v2(&mut host_ptr, byte_size.into()) };
        if result != CUDA_SUCCESS {
            return Err(cust::error::CudaError::UnknownError);
        }

        // SAFETY: The device buffer is not read before being written; all
        // device reads are preceded by `copy_to_device`, which is
        // itself unsafe and requires the caller to synchronize the
        // stream before reading.
        let device_buf = match unsafe {
            cust::memory::DeviceBuffer::uninitialized(len.get())
        } {
            Ok(buf) => buf,
            Err(e) => {
                // SAFETY: `host_ptr` was successfully allocated above and has
                // not been freed.
                unsafe { cuMemFreeHost(host_ptr) };
                return Err(e);
            }
        };

        Ok(PinnedStagingBuffer {
            host_ptr: host_ptr as *mut T,
            device_buf,
            len,
            cuda_device,
        })
    }

    /// View the host staging area as a slice.
    ///
    /// # Safety
    ///
    /// All elements must have been initialized before calling this method.
    pub unsafe fn host(&self) -> &[T] {
        // SAFETY: `host_ptr` is valid for `len` elements (allocated in `new`),
        // lives as long as `self`, and the caller guarantees all elements are
        // initialized per the `# Safety` contract.
        unsafe { std::slice::from_raw_parts(self.host_ptr, self.len.get()) }
    }

    /// View the host staging area as a mutable slice for writing from the CPU.
    ///
    /// # Safety
    ///
    /// All elements must have been initialized before calling this method.
    pub unsafe fn host_mut(&mut self) -> &mut [T] {
        // SAFETY: Same as `host`; unique access is guaranteed by `&mut self`.
        unsafe { std::slice::from_raw_parts_mut(self.host_ptr, self.len.get()) }
    }

    /// Enqueue an async H2D copy on `stream` using the DMA copy engine.
    ///
    /// # Safety
    ///
    /// The host buffer must not be modified and the device buffer must not be
    /// read until `stream` is synchronized.
    pub unsafe fn copy_to_device(
        &mut self,
        stream: &cust::stream::Stream,
    ) -> cust::error::CudaResult<()> {
        use cust::memory::AsyncCopyDestination;
        // SAFETY: `host_slice` points to the pinned allocation valid for `len`
        // reads; the caller guarantees no concurrent modification per `#
        // Safety`.
        unsafe {
            let host_slice =
                std::slice::from_raw_parts(self.host_ptr, self.len.get());
            self.device_buf.async_copy_from(host_slice, stream)
        }
    }

    /// Enqueue an async D2H copy on `stream` using the DMA copy engine.
    ///
    /// # Safety
    ///
    /// The device buffer must not be modified and the host buffer must not be
    /// read until `stream` is synchronized.
    pub unsafe fn copy_to_host(
        &mut self,
        stream: &cust::stream::Stream,
    ) -> cust::error::CudaResult<()> {
        use cust::memory::AsyncCopyDestination;
        // SAFETY: `host_slice` points to the pinned allocation valid for `len`
        // writes; the caller guarantees no concurrent reads per `# Safety`.
        unsafe {
            let host_slice =
                std::slice::from_raw_parts_mut(self.host_ptr, self.len.get());
            self.device_buf.async_copy_to(host_slice, stream)
        }
    }

    pub fn len(&self) -> NonZeroUsize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        false
    }
}

impl<T: UniversalCopy> Drop for PinnedStagingBuffer<T> {
    fn drop(&mut self) {
        let _context = Device::CUDA(self.cuda_device).get_context();
        // SAFETY: `host_ptr` was allocated by `cuMemAllocHost_v2` in `new` and
        // has not been freed; `device_buf` is freed by its own Drop impl.
        unsafe {
            cust_raw::cuMemFreeHost(self.host_ptr as *mut std::os::raw::c_void);
        }
    }
}

impl<T: UniversalCopy> AsUPtr<T> for PinnedStagingBuffer<T> {
    fn as_uptr(&self, device: Device) -> *const T {
        match device {
            Device::CPU => self.host_ptr as *const T,
            Device::CUDA(c) => {
                assert_eq!(
                    c, self.cuda_device,
                    "requested ptr for device {c} but allocated for {}",
                    self.cuda_device
                );
                self.device_buf.as_device_ptr().as_ptr()
            }
        }
    }
}

impl<T: UniversalCopy> AsUPtrMut<T> for PinnedStagingBuffer<T> {
    fn as_mut_uptr(&mut self, device: Device) -> *mut T {
        match device {
            Device::CPU => self.host_ptr,
            Device::CUDA(c) => {
                assert_eq!(
                    c, self.cuda_device,
                    "requested ptr for device {c} but allocated for {}",
                    self.cuda_device
                );
                self.device_buf.as_device_ptr().as_mut_ptr()
            }
        }
    }
}

/// A pinned host buffer mapped into the CUDA device address space.
///
/// Allocated with `cuMemHostAlloc(CU_MEMHOSTALLOC_DEVICEMAP |
/// CU_MEMHOSTALLOC_PORTABLE)`.
///
/// Holds both a host pointer (for CPU access) and a device pointer obtained via
/// `cuMemHostGetDevicePointer_v2` (for GPU access without a copy).
///
/// # When to use
///
/// This enables zero-copy GPU reads from host memory suitable when copying via
/// PCI-e. The tradeoff is that access latency is higher than device DRAM; this
/// is best suited for data that is written once on the host and read
/// infrequently on the device.
///
/// # Multi-device usage
///
/// Only the CUDA device specified at construction is supported (and asserted)
/// via [`AsUPtr`]/[`AsUPtrMut`]. With the `PORTABLE` flag, the physical mapping
/// is accessible from all CUDA contexts, but CUDA does not guarantee the
/// virtual device address is identical across contexts. Callers needing
/// multi-device access should construct one buffer per device.
///
/// # Note
///
/// Memory is not initialized on allocation; callers must write before reading.
pub struct PinnedMappedBuffer<T: UniversalCopy> {
    host_ptr: *mut T,
    device_ptr: *mut T,
    len: NonZeroUsize,
    // TODO should be Device maybe? we dont allow cpu-cpu here? although it
    // would be nice to hide this abstraction here (cpu-cpu would just be
    // shared memory access or a channel?)
    cuda_device: u8,
}

// SAFETY: `PinnedMappedBuffer` owns its pinned host allocation; no
// unsynchronised aliasing exists across thread boundaries.
unsafe impl<T: UniversalCopy> Send for PinnedMappedBuffer<T> {}
// SAFETY: Shared access is safe because the host buffer is only mutated when
// the caller has exclusive access (enforced by `&mut self`).
unsafe impl<T: UniversalCopy> Sync for PinnedMappedBuffer<T> {}

impl<T: UniversalCopy> PinnedMappedBuffer<T> {
    /// Allocate `len` elements of pinned host memory mapped to `cuda_device`.
    ///
    /// Memory is uninitialized. Returns an error if the CUDA allocation fails.
    pub fn new(
        len: NonZeroUsize,
        cuda_device_id: u8,
    ) -> cust::error::CudaResult<Self> {
        use std::os::raw::c_void;

        use cust_raw::{
            CU_MEMHOSTALLOC_DEVICEMAP, CU_MEMHOSTALLOC_PORTABLE, cuMemFreeHost,
            cuMemHostAlloc, cuMemHostGetDevicePointer_v2,
            cudaError_enum::CUDA_SUCCESS,
        };

        // Check at compile time instead of error during allocation
        const {
            assert!(
                std::mem::size_of::<T>() > 0,
                "T must not be a zero-sized type"
            );
        }

        // We know due to the compile time assert that sizeof T is nonzero
        let size_of_t = NonZeroUsize::new(std::mem::size_of::<T>()).unwrap();
        let byte_size = len
            .checked_mul(size_of_t)
            .ok_or(cust::error::CudaError::InvalidMemoryAllocation)?;

        let _context = Device::CUDA(cuda_device_id).get_context();

        let mut host_ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: `host_ptr` is a valid out-pointer and `byte_size` is
        // non-zero.
        let result = unsafe {
            cuMemHostAlloc(
                &mut host_ptr,
                byte_size.into(),
                // TODO remove PORTABLE?
                CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE,
            )
        };
        if result != CUDA_SUCCESS {
            return Err(cust::error::CudaError::UnknownError);
        }

        let mut device_ptr_raw: cust_raw::CUdeviceptr = 0;
        // SAFETY: `host_ptr` is a valid pointer to pinned memory allocated
        // above; `device_ptr_raw` is a valid out-pointer; flags must be 0.
        let result = unsafe {
            cuMemHostGetDevicePointer_v2(&mut device_ptr_raw, host_ptr, 0)
        };
        if result != CUDA_SUCCESS {
            // SAFETY: `host_ptr` was successfully allocated above and has not
            // been freed.
            unsafe { cuMemFreeHost(host_ptr) };
            return Err(cust::error::CudaError::UnknownError);
        }

        Ok(PinnedMappedBuffer {
            host_ptr: host_ptr as *mut T,
            device_ptr: device_ptr_raw as *mut T,
            len,
            cuda_device: cuda_device_id,
        })
    }

    pub fn len(&self) -> NonZeroUsize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        false
    }
}

impl<T: UniversalCopy> Drop for PinnedMappedBuffer<T> {
    fn drop(&mut self) {
        let _context = Device::CUDA(self.cuda_device).get_context();
        // SAFETY: `host_ptr` was allocated by `cuMemHostAlloc` in `new` and has
        // not been freed.
        unsafe {
            cust_raw::cuMemFreeHost(self.host_ptr as *mut std::os::raw::c_void);
        }
    }
}

impl<T: UniversalCopy> AsUPtr<T> for PinnedMappedBuffer<T> {
    fn as_uptr(&self, device: Device) -> *const T {
        match device {
            Device::CPU => self.host_ptr as *const T,
            Device::CUDA(c) => {
                assert_eq!(
                    c, self.cuda_device,
                    "requested ptr for device {c} but allocated for {}",
                    self.cuda_device
                );
                self.device_ptr as *const T
            }
        }
    }
}

impl<T: UniversalCopy> AsUPtrMut<T> for PinnedMappedBuffer<T> {
    fn as_mut_uptr(&mut self, device: Device) -> *mut T {
        match device {
            Device::CPU => self.host_ptr,
            Device::CUDA(c) => {
                assert_eq!(
                    c, self.cuda_device,
                    "requested ptr for device {c} but allocated for {}",
                    self.cuda_device
                );
                self.device_ptr
            }
        }
    }
}
