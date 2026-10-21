// SPDX-FileCopyrightText: Copyright (c) Zizheng Guo. Originally from eda-infra-rs (https://github.com/gzz2000/eda-infra-rs).
// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH. Modifications to the original file.
// SPDX-License-Identifier: Apache-2.0
//! Memory usage profiling utilities

#[cfg(feature = "cuda")]
use cust_raw::cuMemGetInfo_v2;
use memory_stats::memory_stats;
use size::Size;

use crate::Device;

/// Returns CPU memory usage of current process.
///
/// the memory is the size of the physical residence set (working set).
/// obtained using `memory_stats` crate.
///
/// note this is NOT the peak.
pub fn cpu_physical_mem() -> usize {
    memory_stats().unwrap().physical_mem
}

/// Returns CUDA device used memory and total memory.
///
/// obtained using [cuMemGetInfo](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MEM.html#group__CUDA__MEM_1g808f555540d0143a331cc42aa98835c0).
///
/// not this is NOT the peak, and also NOT only the memory used
/// by current process but other processes as well.
#[cfg(feature = "cuda")]
pub fn cuda_used_total_mem(cuda_device_id: u8) -> (usize, usize) {
    let (mut free, mut total) = (0, 0);
    let _context = Device::CUDA(cuda_device_id).get_context();
    // SAFETY: The CUDA context for `cuda_device_id` was just activated above;
    // `free` and `total` are valid stack-allocated output parameters.
    unsafe {
        let ret = cuMemGetInfo_v2(&mut free, &mut total);
        assert_eq!(ret, cust_raw::cudaError_enum::CUDA_SUCCESS);
    }
    (total - free, total)
}

/// Get memory used on the specified universal device.
pub fn device_mem_used(device: Device) -> usize {
    match device {
        Device::CPU => cpu_physical_mem(),
        #[cfg(feature = "cuda")]
        Device::CUDA(cuid) => cuda_used_total_mem(cuid).0,
    }
}

/// Print in the logging interface the memory usage of all
/// managed heterogeneous devices.
pub fn log_memory_stats() {
    let cpu_mem = cpu_physical_mem();
    tracing::info!("cpu memory footprint: {}", Size::from_bytes(cpu_mem));
    #[cfg(feature = "cuda")]
    {
        for cuid in 0..*crate::NUM_CUDA_DEVICES {
            let (cuda_used, cuda_total) =
                cuda_used_total_mem(cuid.try_into().unwrap());
            tracing::info!(
                "cuda device {} memory usage: {} / {}",
                cuid,
                Size::from_bytes(cuda_used),
                Size::from_bytes(cuda_total)
            );
        }
    }
}
