//! Fault profiles.

use esp_idf_svc::sys::esp_random;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaultMode {
    Normal,
    Spike,
    Drift,
    Stuck,
    All,
}

impl FaultMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Spike => "spike",
            Self::Drift => "drift",
            Self::Stuck => "stuck",
            Self::All => "all",
        }
    }

    pub fn is_software_injected(self) -> bool {
        matches!(self, Self::Spike | Self::Drift | Self::Stuck | Self::All)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FaultConfig {
    pub mode: FaultMode,
    pub read_pin: i32,
    pub bypass_checksum: bool,
    pub baseline_sigma: f32,
    pub spike_magnitude_sigma_range: (f32, f32),
    pub spike_magnitude_range: (f32, f32),
    pub drift_rate_sigma_range: (f32, f32),
    pub drift_rate_range: (f32, f32),
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
    mode: FaultMode,
    emitted_mode: FaultMode,
    step: u64,
    event_offset: f32,
    stuck_value: f32,
}

impl FaultInjector {
    pub fn new(config: FaultConfig) -> Self {
        let mode = if config.mode == FaultMode::All {
            FaultMode::Normal
        } else {
            config.mode
        };
        Self {
            config,
            mode,
            emitted_mode: mode,
            step: 0,
            event_offset: 0.0,
            stuck_value: 0.0,
        }
    }

    pub fn emitted_mode(&self) -> FaultMode {
        self.emitted_mode
    }

    pub fn apply(&mut self, temperature: f32) -> f32 {
        self.emitted_mode = self.mode;
        if self.step == 0 {
            match self.mode {
                FaultMode::Spike => {
                    self.event_offset = self.sample_signed(
                        self.config.spike_magnitude_sigma_range,
                        self.config.spike_magnitude_range,
                    );
                }
                FaultMode::Drift => {
                    self.event_offset = self.sample_signed(
                        self.config.drift_rate_sigma_range,
                        self.config.drift_rate_range,
                    );
                }
                FaultMode::Stuck => self.stuck_value = temperature,
                FaultMode::Normal => (),
                FaultMode::All => unreachable!(),
            }
        }
        self.step = self.step.saturating_add(1);
        let output = match self.mode {
            FaultMode::Normal => temperature,
            FaultMode::Spike => temperature + self.event_offset,
            FaultMode::Drift => temperature + self.event_offset * self.step as f32,
            FaultMode::Stuck => self.stuck_value,
            FaultMode::All => unreachable!(),
        };
        if self.config.mode == FaultMode::All {
            let next = self.next_mode();
            if next != self.mode {
                self.step = 0;
            }
            self.mode = next;
        } else if self.step >= u64::from(self.config.event_duration_samples) {
            self.step = 0;
        }
        output
    }

    fn sample_signed(&self, sigma_range: (f32, f32), fallback: (f32, f32)) -> f32 {
        let (range, scale) = if self.config.baseline_sigma > 0.0 {
            (sigma_range, self.config.baseline_sigma)
        } else {
            (fallback, 1.0)
        };
        let magnitude = ((range.0 as f64 + random_uniform() * (range.1 as f64 - range.0 as f64))
            * scale as f64) as f32;
        if (unsafe { esp_random() }) & 1 == 1 {
            magnitude
        } else {
            -magnitude
        }
    }

    fn next_mode(&self) -> FaultMode {
        let draw = random_uniform();
        match self.mode {
            FaultMode::Normal => {
                if draw < 0.017647 {
                    FaultMode::Spike
                } else if draw < 0.017647 + 0.003971 {
                    FaultMode::Drift
                } else if draw < 0.017647 + 0.003971 + 0.006176 {
                    FaultMode::Stuck
                } else {
                    FaultMode::Normal
                }
            }
            FaultMode::Spike if draw < 1.0 / 2.0 => FaultMode::Normal,
            FaultMode::Drift if draw < 1.0 / 20.0 => FaultMode::Normal,
            FaultMode::Stuck if draw < 1.0 / 10.0 => FaultMode::Normal,
            mode => mode,
        }
    }
}

fn random_uniform() -> f64 {
    unsafe { esp_random() as f64 / (u32::MAX as f64 + 1.0) }
}
