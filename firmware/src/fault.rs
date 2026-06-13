#[derive(Clone, Copy, Debug)]
pub enum FaultMode {
    Normal,
    Spike,
}

impl FaultMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::Spike => "spike",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FaultConfig {
    pub mode: FaultMode,
    pub read_pin: i32,
    pub bypass_checksum: bool,
}

impl FaultConfig {
    pub fn checksum_enabled(self) -> bool {
        !self.bypass_checksum
    }

    pub fn uses_fault_read_pin(self, normal_pin: i32) -> bool {
        self.read_pin != normal_pin
    }
}
