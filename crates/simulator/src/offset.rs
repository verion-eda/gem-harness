// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
use std::fmt::{self, Display};

/// A bit offset into a packed `u32` state array.
#[derive(Copy, Clone, Debug)]
pub struct StateOffset(u32);

impl Display for StateOffset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "offset {} (word={}, bit={})",
            self.0,
            self.word_idx(),
            self.bit_idx()
        )
    }
}

impl From<u32> for StateOffset {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<&StateOffset> for u32 {
    fn from(value: &StateOffset) -> Self {
        value.0
    }
}

impl StateOffset {
    /// Index of the `u32` word containing this bit.
    pub fn word_idx(&self) -> usize {
        // equivalent to self.0 / 32
        (self.0 >> 5) as usize
    }

    /// Bit position within the containing word (0..32).
    pub fn bit_idx(&self) -> u32 {
        // equivalent to self.0 % 32
        self.0 & 31
    }

    /// Read this bit from the given state slice. Returns 0 or 1.
    pub fn read(&self, state: &[u32]) -> u32 {
        state[self.word_idx()] >> self.bit_idx() & 1
    }
}
