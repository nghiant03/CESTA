use esp_idf_svc::wifi::{BlockingWifi, ClientConfiguration, Configuration, EspWifi};
use esp_idf_svc::eventloop::EspSystemEventLoop;
use esp_idf_svc::nvs::EspDefaultNvsPartition;
use esp_idf_hal::modem::Modem;
use log::info;

use crate::config;

pub fn connect() -> BlockingWifi<EspWifi<'static>> {
    let sys_loop = EspSystemEventLoop::take().unwrap();
    let nvs = EspDefaultNvsPartition::take().unwrap();

    let modem = unsafe { Modem::new() };
    let esp_wifi = EspWifi::new(modem, sys_loop.clone(), Some(nvs)).unwrap();
    let mut wifi = BlockingWifi::wrap(esp_wifi, sys_loop).unwrap();

    wifi.set_configuration(&Configuration::Client(ClientConfiguration {
        ssid: config::WIFI_SSID.try_into().unwrap(),
        password: config::WIFI_PASSWORD.try_into().unwrap(),
        ..Default::default()
    }))
    .unwrap();

    info!("[Wifi] Connecting to {}", config::WIFI_SSID);
    wifi.start().unwrap();

    let max_retries = 5;
    for attempt in 1..=max_retries {
        info!("[Wifi] Connection attempt {}/{}", attempt, max_retries);
        match wifi.connect() {
            Ok(_) => break,
            Err(e) => {
                log::warn!("[Wifi] Attempt {} failed: {:?}", attempt, e);
                if attempt == max_retries {
                    panic!("[Wifi] Failed to connect after {} attempts", max_retries);
                }
                std::thread::sleep(std::time::Duration::from_secs(2));
            }
        }
    }
    wifi.wait_netif_up().unwrap();

    let ip_info = wifi.wifi().sta_netif().get_ip_info().unwrap();
    info!(
        "[Wifi] Connected, IP {}, gateway {}, netmask {}, DNS {:?}, secondary DNS {:?}",
        ip_info.ip,
        ip_info.subnet.gateway,
        ip_info.subnet.mask,
        ip_info.dns,
        ip_info.secondary_dns
    );

    wifi
}
