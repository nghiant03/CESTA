use std::{env, path::PathBuf, process::Command};

fn main() {
    embuild::espidf::sysenv::output();
    println!("cargo:rustc-check-cfg=cfg(hifinet)");
    for key in ["FIRMWARE_MODEL", "HIFINET_BUNDLE", "HIFINET_RECEIVER"] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    match env::var("FIRMWARE_MODEL").as_deref().unwrap_or("cesta") {
        "cesta" => (),
        "hifinet" => {
            let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
            let bundle =
                PathBuf::from(env::var_os("HIFINET_BUNDLE").expect("HIFINET_BUNDLE required"))
                    .canonicalize()
                    .unwrap();
            let receiver = env::var("HIFINET_RECEIVER").expect("HIFINET_RECEIVER required");
            let script = root.join("../scripts/install_hifinet.py");
            println!("cargo:rerun-if-changed={}", script.display());
            println!("cargo:rerun-if-changed={}", bundle.display());
            let status = Command::new("uv")
                .args(["run", "--no-project", "python"])
                .arg(script)
                .arg("--bundle")
                .arg(bundle)
                .arg("--receiver")
                .arg(receiver)
                .arg("--rust-output")
                .arg(PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("hifinet_bundle.rs"))
                .status()
                .expect("run HiFiNet bundle validator");
            assert!(status.success(), "HiFiNet bundle validation failed");
            println!("cargo:rustc-cfg=hifinet");
        }
        other => panic!("unsupported FIRMWARE_MODEL: {other}"),
    }
}
