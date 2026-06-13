use esp_idf_hal::delay::Ets;
use esp_idf_hal::gpio::{AnyIOPin, IOPin, InputOutput, PinDriver, Pull};

#[derive(Clone, Copy, Debug)]
pub struct DhtReading {
    pub temperature: f32,
    pub humidity: f32,
}

pub struct Dht11Sensor<'a> {
    pin: PinDriver<'a, AnyIOPin, InputOutput>,
}

impl<'a> Dht11Sensor<'a> {
    pub fn new(pin: impl IOPin + 'a) -> Self {
        let mut driver =
            PinDriver::input_output_od(pin.downgrade()).expect("DHT: Failed to init GPIO");
        driver.set_pull(Pull::Up).expect("DHT: Failed to set pull-up");
        driver.set_high().expect("DHT: Failed to set high");
        Self { pin: driver }
    }

    pub fn read(&mut self, checksum_enabled: bool) -> Result<DhtReading, DhtError> {
        let mut data = [0u8; 5];

        self.pin.set_high().map_err(|_| DhtError::Gpio)?;
        Ets::delay_ms(2);
        self.pin.set_low().map_err(|_| DhtError::Gpio)?;
        Ets::delay_ms(20);
        self.pin.set_high().map_err(|_| DhtError::Gpio)?;
        Ets::delay_us(40);

        if let Err(e) = self.wait_low(200) {
            log::warn!(
                "DHT debug: Step 1 fail (sensor never pulled LOW). \
                 Pin is {}. Sensor disconnected or dead.",
                if self.pin.is_high() { "HIGH" } else { "LOW" }
            );
            return Err(e);
        }
        if let Err(e) = self.wait_high(200) {
            log::warn!("DHT debug: Step 2 fail (sensor never released HIGH).");
            return Err(e);
        }
        if let Err(e) = self.wait_low(200) {
            log::warn!("DHT debug: Step 3 fail (no first data-bit LOW).");
            return Err(e);
        }

        for byte in &mut data {
            for bit in (0..8).rev() {
                self.wait_high(120)?;
                let high_us = self.measure_high(120)?;
                if high_us > 40 {
                    *byte |= 1 << bit;
                }
            }
        }

        let sum = data[0] as u16 + data[1] as u16 + data[2] as u16 + data[3] as u16;
        if checksum_enabled && (sum & 0xFF) != data[4] as u16 {
            return Err(DhtError::Checksum);
        }

        Ok(DhtReading {
            humidity: data[0] as f32 + data[1] as f32 * 0.1,
            temperature: data[2] as f32 + (data[3] & 0x7F) as f32 * 0.1,
        })
    }

    fn wait_low(&self, timeout_us: u32) -> Result<(), DhtError> {
        for _ in 0..timeout_us {
            if self.pin.is_low() {
                return Ok(());
            }
            Ets::delay_us(1);
        }
        Err(DhtError::Timeout)
    }

    fn wait_high(&self, timeout_us: u32) -> Result<(), DhtError> {
        for _ in 0..timeout_us {
            if self.pin.is_high() {
                return Ok(());
            }
            Ets::delay_us(1);
        }
        Err(DhtError::Timeout)
    }

    fn measure_high(&self, timeout_us: u32) -> Result<u32, DhtError> {
        let mut count = 0;
        for _ in 0..timeout_us {
            if self.pin.is_low() {
                return Ok(count);
            }
            Ets::delay_us(1);
            count += 1;
        }
        Err(DhtError::Timeout)
    }
}

#[derive(Debug)]
pub enum DhtError {
    Timeout,
    Checksum,
    Gpio,
}
