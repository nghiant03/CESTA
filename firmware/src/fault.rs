//! Fault profiles.

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

    pub fn is_software_injected(self) -> bool {
        matches!(self, Self::Spike | Self::Drift | Self::Stuck)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FaultConfig {
    pub mode: FaultMode,
    pub read_pin: i32,
    pub bypass_checksum: bool,
    pub spike_magnitude_range: (f32, f32),
    pub drift_rate: f32,
    pub jitter_std: f32,
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

    pub fn apply(&mut self, temperature: f32) -> f32 {
        match self.config.mode {
            FaultMode::Spike => self.apply_spike(temperature),
            FaultMode::Drift => self.apply_drift(temperature),
            FaultMode::Stuck => self.apply_stuck(temperature),
            FaultMode::Normal => temperature,
        }
    }

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

fn standard_normal() -> f32 {
    (0..12).map(|_| random_uniform()).sum::<f32>() - 6.0
}
