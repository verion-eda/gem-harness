// SPDX-FileCopyrightText: Copyright (c) Zizheng Guo. Originally from eda-infra-rs (https://github.com/gzz2000/eda-infra-rs).
// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH. Modifications to the original file.
// SPDX-License-Identifier: Apache-2.0
//! Wrapper of raw pointers that exports universal pointer interface.
//!
//! This is essential in the interaction between C-FFIs and universal
//! functions.
//!
//! The pointer wrappers can only be created unsafely, because the
//! safety of pointers should be ensured outside.

use super::*;

/// A raw immutable pointer wrapper associated with a device.
#[derive(Debug, Copy, Clone)]
pub struct RawUPtr<T: UniversalCopy> {
    ptr: *const T,
    device: Device,
}

/// A raw mutable pointer wrapper associated with a device.
#[derive(Debug)]
pub struct RawUPtrMut<T: UniversalCopy> {
    ptr: *mut T,
    device: Device,
}

impl<T: UniversalCopy> RawUPtr<T> {
    /// # Safety
    ///
    /// `ptr` must be a valid pointer for reads on `device` and must remain
    /// valid for the lifetime of this `RawUPtr`.
    #[inline]
    pub unsafe fn new(ptr: *const T, device: Device) -> Self {
        RawUPtr { ptr, device }
    }
}

impl<T: UniversalCopy> RawUPtrMut<T> {
    /// # Safety
    ///
    /// `ptr` must be a valid, exclusively-owned pointer for reads and writes
    /// on `device` and must remain valid for the lifetime of this `RawUPtrMut`.
    #[inline]
    pub unsafe fn new(ptr: *mut T, device: Device) -> Self {
        RawUPtrMut { ptr, device }
    }
}

impl<T: UniversalCopy> AsUPtr<T> for RawUPtr<T> {
    #[inline]
    fn as_uptr(&self, device: Device) -> *const T {
        assert_eq!(device, self.device, "raw uptr not on requested device");
        self.ptr
    }
}

impl<T: UniversalCopy> AsUPtr<T> for RawUPtrMut<T> {
    #[inline]
    fn as_uptr(&self, device: Device) -> *const T {
        assert_eq!(device, self.device, "raw uptr not on requested device");
        self.ptr
    }
}

impl<T: UniversalCopy> AsUPtrMut<T> for RawUPtrMut<T> {
    #[inline]
    fn as_mut_uptr(&mut self, device: Device) -> *mut T {
        assert_eq!(device, self.device, "raw uptr not on requested device");
        self.ptr
    }
}

/// An explicit null universal pointer.
///
/// To create one, use [`NullUPtr::new`].
///
/// You should only use it in functions that explicitly accepts
/// null pointers (e.g., to represent optional inputs).
#[non_exhaustive]
#[derive(Debug, Copy, Clone)]
pub struct NullUPtr {}

impl NullUPtr {
    /// Create a null universal pointer.
    ///
    /// # Safety
    ///
    /// The caller must only pass this to functions that explicitly accept null
    /// pointers. Passing it to any function that dereferences the pointer is
    /// undefined behaviour.
    pub const unsafe fn new() -> NullUPtr {
        NullUPtr {}
    }

    /// Create reference to a null universal pointer.
    ///
    /// # Safety
    ///
    /// Same as [`NullUPtr::new`]: the returned reference must only be passed to
    /// functions that explicitly accept null pointers.
    pub const unsafe fn new_ref() -> &'static NullUPtr {
        &NullUPtr {}
    }

    /// Create mutable reference to a null universal pointer.
    ///
    /// # Safety
    ///
    /// Same as [`NullUPtr::new`]: the returned reference must only be passed to
    /// functions that explicitly accept null pointers. Additionally, the caller
    /// must ensure no other reference to `NULL_UPTR` exists simultaneously
    /// (standard aliasing rules for `&mut`).
    pub unsafe fn new_mut() -> &'static mut NullUPtr {
        static mut NULL_UPTR: NullUPtr = NullUPtr {};
        // SAFETY: `addr_of_mut!` avoids creating an intermediate reference to
        // the `static mut`; the caller is required (see `# Safety`) to ensure
        // no aliasing `&mut` exists at the same time.
        unsafe { &mut *std::ptr::addr_of_mut!(NULL_UPTR) }
    }
}

impl<T: UniversalCopy> AsUPtr<T> for NullUPtr {
    #[inline]
    fn as_uptr(&self, _device: Device) -> *const T {
        std::ptr::null()
    }
}

impl<T: UniversalCopy> AsUPtrMut<T> for NullUPtr {
    #[inline]
    fn as_mut_uptr(&mut self, _device: Device) -> *mut T {
        std::ptr::null_mut()
    }
}
