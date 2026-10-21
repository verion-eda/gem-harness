// SPDX-FileCopyrightText: Copyright (c) 2026 Verion GmbH
// SPDX-License-Identifier: Apache-2.0
use std::{
    fmt::{self, Display},
    str::FromStr,
};

use compact_str::CompactString;
use netlistdb::HierName;

#[derive(Debug)]
pub enum SignalError {
    InvalidPath(String),
    InvalidBusIndex(String),
    EmptyPath,
}

impl Display for SignalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidPath(err) => {
                write!(f, "invalid signal path: {err}")
            }
            Self::InvalidBusIndex(err) => {
                write!(f, "invalid bus index: {err}")
            }
            Self::EmptyPath => {
                write!(f, "signal path is empty")
            }
        }
    }
}

impl From<SignalError> for anyhow::Error {
    fn from(value: SignalError) -> Self {
        anyhow::anyhow!("{value}")
    }
}

/// A concrete reference to a signal in the design.
///
/// Stores the full dotted path as a flat segment list. The split between cell
/// instance hierarchy and pin name (required by `NetlistDB` internals) is
/// deferred to lookup time: the last segment is the pin name, all preceding
/// segments form the [`HierName`].
///
/// # Examples
///
/// ```
/// use simulator::signals::SignalPath;
///
/// let path: SignalPath = "top.cpu.alu.result".parse().unwrap();
/// let path_with_idx: SignalPath = "top.cpu.alu.result[3]".parse().unwrap();
/// let path_programmatic =
///     SignalPath::new(["top", "cpu", "reg_0"], Some(5)).unwrap();
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SignalPath {
    /// Full path segments, with the pin name as the last element.
    segments: Vec<CompactString>,
    /// Bus bit index, if specified.
    bus_idx: Option<isize>,
}

impl SignalPath {
    /// Construct a `SignalPath` from an iterator of path segments and an
    /// optional bus bit index.
    ///
    /// Returns an error if `segments` is empty or any segment is empty.
    pub fn new(
        segments: impl IntoIterator<Item = impl Into<CompactString>>,
        bus_idx: Option<isize>,
    ) -> Result<Self, SignalError> {
        let segments: Vec<CompactString> =
            segments.into_iter().map(Into::into).collect();
        if segments.is_empty() {
            return Err(SignalError::EmptyPath);
        }
        if segments.iter().any(|s| s.is_empty()) {
            return Err(SignalError::EmptyPath);
        }
        Ok(Self { segments, bus_idx })
    }

    /// Returns the cell instance hierarchy — all segments except the last.
    ///
    /// Returns `HierName::empty()` for top-level signals that have no
    /// enclosing module instance.
    pub fn hier_name(&self) -> HierName {
        HierName::from_topdown_hier_iter(
            self.segments[..self.segments.len() - 1]
                .iter()
                .map(CompactString::as_str),
        )
    }

    /// Returns the pin name — the last path segment.
    pub fn pin_name(&self) -> &CompactString {
        // segments is always non-empty, enforced by constructors.
        &self.segments[self.segments.len() - 1]
    }

    /// Returns the bus bit index, if one was specified.
    pub fn bus_idx(&self) -> Option<isize> {
        self.bus_idx
    }
}

impl FromStr for SignalPath {
    type Err = SignalError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.is_empty() {
            return Err(SignalError::EmptyPath);
        }

        let (name_part, bus_idx) = if let Some(open) = s.rfind('[') {
            let close = s.rfind(']').ok_or_else(|| {
                SignalError::InvalidPath(format!(
                    "missing closing ']' in signal path '{}'",
                    s
                ))
            })?;
            if close < open {
                return Err(SignalError::InvalidPath(format!(
                    "malformed bus index in signal path '{}'",
                    s
                )));
            }
            let idx_str = &s[open + 1..close];
            let idx = idx_str.parse::<isize>().map_err(|_| {
                SignalError::InvalidBusIndex(format!(
                    "invalid bus index '{}' in signal path '{}'",
                    idx_str, s
                ))
            })?;
            (&s[..open], Some(idx))
        } else {
            (s, None)
        };

        Self::new(name_part.split('.'), bus_idx)
            .map_err(|_| SignalError::EmptyPath)
    }
}

impl fmt::Display for SignalPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, seg) in self.segments.iter().enumerate() {
            if i > 0 {
                f.write_str(".")?;
            }
            fmt::Display::fmt(seg, f)?;
        }
        if let Some(idx) = self.bus_idx {
            write!(f, "[{}]", idx)?;
        }
        Ok(())
    }
}

/// A pattern for querying signals by name.
pub enum SignalPattern {
    /// Match exactly one signal by its full path.
    Exact(SignalPath),
    /// Match signals using glob syntax.
    ///
    /// `*` matches any characters within a single path segment.
    /// `**` matches any characters across segment boundaries.
    Glob(String),
    /// Match all signals whose hierarchy starts with the given prefix.
    Prefix(HierName),
}

impl SignalPattern {
    /// Construct a `Prefix` pattern from a dotted hierarchy string,
    /// e.g. `"top.cpu.alu"`.
    pub fn prefix(s: &str) -> Result<Self, String> {
        if s.is_empty() {
            return Err("prefix must not be empty".to_string());
        }
        let segments: Vec<&str> = s.split('.').collect();
        if segments.iter().any(|seg| seg.is_empty()) {
            return Err(format!("empty segment in prefix '{}'", s));
        }
        Ok(Self::Prefix(HierName::from_topdown_hier_iter(segments)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_top_level_scalar() {
        let path: SignalPath = "result".parse().unwrap();
        assert_eq!(path.pin_name().as_str(), "result");
        assert!(path.hier_name().is_empty());
        assert_eq!(path.bus_idx(), None);
        assert_eq!(path.to_string(), "result");
    }

    #[test]
    fn parse_hierarchical() {
        let path: SignalPath = "top.cpu.alu.result".parse().unwrap();
        assert_eq!(path.pin_name().as_str(), "result");
        assert_eq!(path.hier_name().to_string(), "top/cpu/alu");
        assert_eq!(path.bus_idx(), None);
        assert_eq!(path.to_string(), "top.cpu.alu.result");
    }

    #[test]
    fn parse_with_bus_idx() {
        let path: SignalPath = "top.cpu.alu.result[3]".parse().unwrap();
        assert_eq!(path.pin_name().as_str(), "result");
        assert_eq!(path.bus_idx(), Some(3));
        assert_eq!(path.to_string(), "top.cpu.alu.result[3]");
    }

    #[test]
    fn parse_top_level_with_bus_idx() {
        let path: SignalPath = "result[0]".parse().unwrap();
        assert!(path.hier_name().is_empty());
        assert_eq!(path.bus_idx(), Some(0));
    }

    #[test]
    fn parse_errors() {
        assert!(SignalPath::from_str("").is_err());
        assert!(SignalPath::from_str("top..cpu").is_err());
        assert!(SignalPath::from_str("top.cpu[abc]").is_err());
        assert!(SignalPath::from_str("top.cpu[3").is_err());
    }

    #[test]
    fn construct_programmatic() {
        let path = SignalPath::new(["top", "cpu", "result"], Some(5)).unwrap();
        assert_eq!(path.to_string(), "top.cpu.result[5]");
    }

    #[test]
    fn prefix_pattern() {
        let pattern = SignalPattern::prefix("top.cpu.alu").unwrap();
        assert!(matches!(pattern, SignalPattern::Prefix(_)));
        assert!(SignalPattern::prefix("").is_err());
        assert!(SignalPattern::prefix("top..cpu").is_err());
    }
}
