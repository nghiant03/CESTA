//! Fault profiles selected by `config::FAULT_CONFIG`, one per fault type in
//! the main codebase (`src/CESTA/schema/fault.py`). SPIKE, DRIFT, and STUCK
//! are software-injected on the normal reading, mirroring the Python
//! injectors with per-event randomization.

use esp_idf_svc::sys::esp_random;

#[derive(Clone, Copy, Debug)]
pub enum FaultMode {
    Normal,
    Spike,
    Drift,
    Stuck,
}

impl FaultMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Spike => "spike",
            Self::Drift => "drift",
            Self::Stuck => "stuck",
        }
    }

    /// All fault modes are software-injected: they transform the normal
    /// sensor reading in firmware.
    pub fn is_software_injected(self) -> bool {
        matches!(self, Self::Spike | Self::Drift | Self::Stuck)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FaultConfig {
    pub mode: FaultMode,
    pub read_pin: i32,
    pub bypass_checksum: bool,
    /// Inclusive (min, max) absolute spike offset in °C, sampled per event
    /// (Spike mode only).
    pub spike_magnitude_range: (f32, f32),
    /// Drift offset added per sample within an event, in °C (Drift mode only).
    pub drift_rate: f32,
    /// Gaussian jitter standard deviation around the frozen value, in °C
    /// (Stuck mode only).
    pub jitter_std: f32,
    /// Samples per fault event before a new event starts (fault modes only).
    pub event_duration_samples: u32,
}

impl FaultConfig {
    pub fn checksum_enabled(self) -> bool {
        !self.bypass_checksum
    }

    pub fn uses_fault_read_pin(self, normal_pin: i32) -> bool {
        self.read_pin != normal_pin
    }
}

/// Software fault injector mirroring the Python injectors in
/// `src/CESTA/injection/faults.py`. Faults are applied as consecutive events
/// of `event_duration_samples` samples; each event re-randomizes its
/// parameters like a new contiguous Markov segment in the Python injectors.
pub struct FaultInjector {
    config: FaultConfig,
    step: u32,
    spike_offset: f32,
    direction: f32,
    stuck_value: f32,
}

impl FaultInjector {
    pub fn new(config: FaultConfig) -> Self {
        Self {
            config,
            step: 0,
            spike_offset: 0.0,
            direction: 1.0,
            stuck_value: 0.0,
        }
    }

    /// Return the faulted temperature for one normal-path sample.
    pub fn apply(&mut self, temperature: f32) -> f32 {
        match self.config.mode {
            FaultMode::Spike => self.apply_spike(temperature),
            FaultMode::Drift => self.apply_drift(temperature),
            FaultMode::Stuck => self.apply_stuck(temperature),
            FaultMode::Normal => temperature,
        }
    }

    /// Spike: constant offset for the whole event, with magnitude sampled
    /// from `spike_magnitude_range` and a random sign per event, mirroring
    /// `SpikeFaultInjector`.
    fn apply_spike(&mut self, temperature: f32) -> f32 {
        if self.step == 0 {
            let (lo, hi) = self.config.spike_magnitude_range;
            let magnitude = lo + random_uniform() * (hi - lo);
            self.spike_offset = if random_bool() { magnitude } else { -magnitude };
        }
        self.step += 1;
        if self.step >= self.config.event_duration_samples {
            self.step = 0;
        }
        temperature + self.spike_offset
    }

    /// Linear drift: `temperature + direction * drift_rate * i` for the
    /// 1-based sample index `i` within the current event, with a random
    /// direction per event, mirroring `DriftFaultInjector`.
    fn apply_drift(&mut self, temperature: f32) -> f32 {
        if self.step == 0 {
            self.direction = if random_bool() { 1.0 } else { -1.0 };
        }
        self.step += 1;
        let offset = self.direction * self.config.drift_rate * self.step as f32;
        if self.step >= self.config.event_duration_samples {
            self.step = 0;
        }
        temperature + offset
    }

    /// Stuck-at-value: freeze at the first reading of each event and add
    /// Gaussian jitter with `jitter_std`, mirroring `StuckFaultInjector`.
    fn apply_stuck(&mut self, temperature: f32) -> f32 {
        if self.step == 0 {
            self.stuck_value = temperature;
        }
        self.step += 1;
        if self.step >= self.config.event_duration_samples {
            self.step = 0;
        }
        self.stuck_value + self.config.jitter_std * standard_normal()
    }
}

fn random_bool() -> bool {
    (unsafe { esp_random() }) & 1 == 1
}

fn random_uniform() -> f32 {
    (unsafe { esp_random() } as f64 / (u32::MAX as f64 + 1.0)) as f32
}

/// Approximate standard normal via Irwin-Hall (sum of 12 uniforms minus 6).
fn standard_normal() -> f32 {
    (0..12).map(|_| random_uniform()).sum::<f32>() - 6.0
}
