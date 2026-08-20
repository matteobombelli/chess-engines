use std::collections::{BTreeMap, HashSet};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::AnalysisRow;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CalibrationConfig {
    pub minimum_rating: u16,
    pub maximum_rating: u16,
    pub bin_width: u16,
    pub minimum_samples_per_bin: usize,
    pub bootstrap_repetitions: usize,
    pub seed: u64,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            minimum_rating: 400,
            maximum_rating: 2_600,
            bin_width: 200,
            minimum_samples_per_bin: 25,
            bootstrap_repetitions: 1_000,
            seed: 1,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RatingBand {
    pub minimum_rating: u16,
    pub maximum_rating: u16,
    pub samples: usize,
    pub games: usize,
    pub human_mean_loss: f64,
    pub bot_mean_loss: f64,
}

impl RatingBand {
    pub fn center(&self) -> f64 {
        (f64::from(self.minimum_rating) + f64::from(self.maximum_rating)) / 2.0
    }

    pub fn human_minus_bot(&self) -> f64 {
        self.human_mean_loss - self.bot_mean_loss
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RatingEstimate {
    Estimated(f64),
    BelowRange(u16),
    AboveRange(u16),
}

#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationReport {
    pub estimate: RatingEstimate,
    pub interval_95: Option<(f64, f64)>,
    pub bootstrap_finite: usize,
    pub slope_per_100_rating: f64,
    pub r_squared: f64,
    pub bands: Vec<RatingBand>,
}

pub fn calibrate(
    rows: &[AnalysisRow],
    config: CalibrationConfig,
) -> Result<CalibrationReport, String> {
    validate_config(config)?;
    let bands = build_bands(rows.iter(), config);
    let fit = fit_bands(&bands).ok_or_else(|| {
        "not enough populated rating bands, or move loss did not improve with rating".to_string()
    })?;
    let estimate = classify_estimate(fit.root, &bands);

    let mut bootstrap = bootstrap_player_estimates(rows, config);
    bootstrap.sort_by(f64::total_cmp);
    let interval_95 = if bootstrap.len() >= 20 {
        Some((percentile(&bootstrap, 0.025), percentile(&bootstrap, 0.975)))
    } else {
        None
    };

    Ok(CalibrationReport {
        estimate,
        interval_95,
        bootstrap_finite: bootstrap.len(),
        slope_per_100_rating: fit.slope * 100.0,
        r_squared: fit.r_squared,
        bands,
    })
}

fn validate_config(config: CalibrationConfig) -> Result<(), String> {
    if config.minimum_rating >= config.maximum_rating {
        return Err("minimum rating must be below maximum rating".to_string());
    }
    if config.bin_width == 0 || config.minimum_samples_per_bin == 0 {
        return Err("bin width and minimum samples must be greater than zero".to_string());
    }
    Ok(())
}

fn build_bands<'a>(
    rows: impl IntoIterator<Item = &'a AnalysisRow>,
    config: CalibrationConfig,
) -> Vec<RatingBand> {
    #[derive(Default)]
    struct Accumulator<'a> {
        rows: Vec<&'a AnalysisRow>,
    }

    let mut grouped: BTreeMap<u16, Accumulator<'_>> = BTreeMap::new();
    for row in rows {
        if !(config.minimum_rating..=config.maximum_rating).contains(&row.actor_rating) {
            continue;
        }
        let offset = row.actor_rating - config.minimum_rating;
        let start = config.minimum_rating + (offset / config.bin_width) * config.bin_width;
        grouped.entry(start).or_default().rows.push(row);
    }

    grouped
        .into_iter()
        .filter(|(_, accumulator)| accumulator.rows.len() >= config.minimum_samples_per_bin)
        .map(|(start, accumulator)| {
            let samples = accumulator.rows.len();
            let games = accumulator
                .rows
                .iter()
                .map(|row| row.game_id.as_str())
                .collect::<HashSet<_>>()
                .len();
            let human_mean_loss = accumulator
                .rows
                .iter()
                .map(|row| row.human_loss)
                .sum::<f64>()
                / samples as f64;
            let bot_mean_loss =
                accumulator.rows.iter().map(|row| row.bot_loss).sum::<f64>() / samples as f64;
            RatingBand {
                minimum_rating: start,
                maximum_rating: start
                    .saturating_add(config.bin_width - 1)
                    .min(config.maximum_rating),
                samples,
                games,
                human_mean_loss,
                bot_mean_loss,
            }
        })
        .collect()
}

#[derive(Clone, Copy)]
struct Fit {
    root: f64,
    slope: f64,
    r_squared: f64,
}

/// Weighted regression of paired loss difference against human rating. The
/// zero crossing is where human and bot expected-point loss are equal.
fn fit_bands(bands: &[RatingBand]) -> Option<Fit> {
    if bands.len() < 2 {
        return None;
    }
    let weight = bands.iter().map(|band| band.samples as f64).sum::<f64>();
    let x_mean = bands
        .iter()
        .map(|band| band.center() * band.samples as f64)
        .sum::<f64>()
        / weight;
    let y_mean = bands
        .iter()
        .map(|band| band.human_minus_bot() * band.samples as f64)
        .sum::<f64>()
        / weight;
    let covariance = bands
        .iter()
        .map(|band| {
            band.samples as f64 * (band.center() - x_mean) * (band.human_minus_bot() - y_mean)
        })
        .sum::<f64>();
    let x_variance = bands
        .iter()
        .map(|band| band.samples as f64 * (band.center() - x_mean).powi(2))
        .sum::<f64>();
    if x_variance == 0.0 {
        return None;
    }
    let slope = covariance / x_variance;
    // Human loss should fall relative to bot loss as rating rises.
    if !slope.is_finite() || slope >= 0.0 {
        return None;
    }
    let intercept = y_mean - slope * x_mean;
    let root = -intercept / slope;

    let residual = bands
        .iter()
        .map(|band| {
            let predicted = intercept + slope * band.center();
            band.samples as f64 * (band.human_minus_bot() - predicted).powi(2)
        })
        .sum::<f64>();
    let total = bands
        .iter()
        .map(|band| band.samples as f64 * (band.human_minus_bot() - y_mean).powi(2))
        .sum::<f64>();
    let r_squared = if total == 0.0 {
        1.0
    } else {
        (1.0 - residual / total).clamp(0.0, 1.0)
    };
    Some(Fit {
        root,
        slope,
        r_squared,
    })
}

fn classify_estimate(root: f64, bands: &[RatingBand]) -> RatingEstimate {
    let low = bands.first().expect("fit requires bands").minimum_rating;
    let high = bands.last().expect("fit requires bands").maximum_rating;
    if root < f64::from(low) {
        RatingEstimate::BelowRange(low)
    } else if root > f64::from(high) {
        RatingEstimate::AboveRange(high)
    } else {
        RatingEstimate::Estimated(root)
    }
}

/// Cluster bootstrap by human player. Repeated moves by one person share style,
/// skill, and rating history, so treating them as independent positions would
/// make the interval too narrow.
fn bootstrap_player_estimates(rows: &[AnalysisRow], config: CalibrationConfig) -> Vec<f64> {
    if config.bootstrap_repetitions == 0 {
        return Vec::new();
    }
    let mut by_player: BTreeMap<String, Vec<&AnalysisRow>> = BTreeMap::new();
    for row in rows {
        by_player
            .entry(row.actor_username.to_ascii_lowercase())
            .or_default()
            .push(row);
    }
    let players: Vec<_> = by_player.into_values().collect();
    if players.is_empty() {
        return Vec::new();
    }

    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut estimates = Vec::with_capacity(config.bootstrap_repetitions);
    for _ in 0..config.bootstrap_repetitions {
        let mut resampled = Vec::new();
        for _ in 0..players.len() {
            let player = &players[rng.gen_range(0..players.len())];
            resampled.extend(player.iter().copied());
        }
        let bands = build_bands(resampled, config);
        if let Some(fit) = fit_bands(&bands)
            && fit.root.is_finite()
        {
            estimates.push(fit.root);
        }
    }
    estimates
}

fn percentile(sorted: &[f64], probability: f64) -> f64 {
    let index = probability * (sorted.len() - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;
    let fraction = index - lower as f64;
    sorted[lower] * (1.0 - fraction) + sorted[upper] * fraction
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(game: usize, rating: u16, human_loss: f64, bot_loss: f64) -> AnalysisRow {
        AnalysisRow {
            game_id: format!("game-{game}"),
            actor_username: format!("player-{game}"),
            actor_rating: rating,
            ply: 20,
            uci_prefix: Vec::new(),
            fen: "fen".to_string(),
            human_move: "e2e4".to_string(),
            bot_move: "d2d4".to_string(),
            reference_move: "g1f3".to_string(),
            best_expected_score: 0.5,
            human_expected_score: 0.5 - human_loss,
            bot_expected_score: 0.5 - bot_loss,
            human_loss,
            bot_loss,
        }
    }

    #[test]
    fn finds_the_equal_quality_rating() {
        let mut rows = Vec::new();
        for (band, rating) in [500, 900, 1_300, 1_700, 2_100].into_iter().enumerate() {
            let human_loss = 0.25 - f64::from(rating) / 10_000.0;
            for sample in 0..30 {
                rows.push(row(band * 100 + sample, rating, human_loss, 0.10));
            }
        }
        let report = calibrate(
            &rows,
            CalibrationConfig {
                minimum_samples_per_bin: 10,
                bootstrap_repetitions: 100,
                ..CalibrationConfig::default()
            },
        )
        .unwrap();
        let RatingEstimate::Estimated(rating) = report.estimate else {
            panic!("expected a finite rating")
        };
        assert!((rating - 1_500.0).abs() < 1.0, "rating was {rating}");
        assert!(report.interval_95.is_some());
        assert!(report.r_squared > 0.99);
    }

    #[test]
    fn requires_at_least_two_populated_bins() {
        let rows = vec![row(1, 1_000, 0.2, 0.1)];
        let error = calibrate(
            &rows,
            CalibrationConfig {
                minimum_samples_per_bin: 1,
                ..CalibrationConfig::default()
            },
        )
        .unwrap_err();
        assert!(error.contains("not enough populated"));
    }
}
