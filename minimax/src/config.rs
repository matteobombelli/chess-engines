use std::time::Duration;

/// Depth, time, and node limits for one search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SearchLimits {
    pub max_depth: u8,
    pub move_time: Option<Duration>,
    pub max_nodes: Option<u64>,
}

impl SearchLimits {
    pub const MAX_SUPPORTED_DEPTH: u8 = 64;

    pub fn fixed_depth(max_depth: u8) -> Result<Self, String> {
        let limits = Self {
            max_depth,
            move_time: None,
            max_nodes: None,
        };
        limits.validate()?;
        Ok(limits)
    }

    /// Read search limits from the environment.
    pub fn from_env() -> Result<Self, String> {
        let max_depth = parse_env("MINIMAX_DEPTH")?.unwrap_or(5);
        let move_time_ms: Option<u64> = parse_env("MINIMAX_MOVE_TIME_MS")?;
        let max_nodes = parse_env("MINIMAX_MAX_NODES")?;

        let limits = Self {
            max_depth,
            move_time: move_time_ms.map(Duration::from_millis),
            max_nodes,
        };
        limits.validate()?;
        Ok(limits)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.max_depth == 0 || self.max_depth > Self::MAX_SUPPORTED_DEPTH {
            return Err(format!(
                "search depth must be between 1 and {}, got {}",
                Self::MAX_SUPPORTED_DEPTH,
                self.max_depth
            ));
        }
        if self.move_time.is_some_and(|time| time.is_zero()) {
            return Err("move time must be greater than zero".to_string());
        }
        if self.max_nodes == Some(0) {
            return Err("node limit must be greater than zero".to_string());
        }
        Ok(())
    }
}

impl Default for SearchLimits {
    fn default() -> Self {
        Self {
            max_depth: 5,
            move_time: None,
            max_nodes: None,
        }
    }
}

fn parse_env<T>(name: &str) -> Result<Option<T>, String>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    match std::env::var(name) {
        Ok(value) => value
            .parse()
            .map(Some)
            .map_err(|error| format!("invalid {name}={value:?}: {error}")),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(format!("could not read {name}: {error}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_depth() {
        assert!(SearchLimits::fixed_depth(0).is_err());
        assert!(SearchLimits::fixed_depth(65).is_err());
        assert_eq!(SearchLimits::fixed_depth(4).unwrap().max_depth, 4);
    }

    #[test]
    fn rejects_zero_limits() {
        let limits = SearchLimits {
            max_depth: 4,
            move_time: Some(Duration::ZERO),
            max_nodes: None,
        };
        assert!(limits.validate().is_err());

        let limits = SearchLimits {
            move_time: None,
            max_nodes: Some(0),
            ..limits
        };
        assert!(limits.validate().is_err());
    }
}
