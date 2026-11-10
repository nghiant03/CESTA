use std::{env, path::PathBuf, process::Command};

fn main() {
    embuild::espidf::sysenv::output();
    println!("cargo:rustc-check-cfg=cfg(hifinet)");
    println!("cargo:rustc-check-cfg=cfg(stgcn)");
    for key in [
        "FIRMWARE_MODEL",
        "HIFINET_BUNDLE",
        "HIFINET_RECEIVER",
        "STGCN_BUNDLE",
        "STGCN_RECEIVER",
    ] {
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
            let script = root.join("../scripts/install_firmware.py");
            println!("cargo:rerun-if-changed={}", script.display());
            println!("cargo:rerun-if-changed={}", bundle.display());
            let status = Command::new("uv")
                .args(["run", "--no-project", "--with", "tflite==2.18.0", "python"])
                .arg(script)
                .args(["--architecture", "hifinet"])
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
        "stgcn" => {
            let root = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
            let bundle = PathBuf::from(env::var_os("STGCN_BUNDLE").expect("STGCN_BUNDLE required"))
                .canonicalize()
                .unwrap();
            let receiver = env::var("STGCN_RECEIVER").expect("STGCN_RECEIVER required");
            let script = root.join("../scripts/install_firmware.py");
            println!("cargo:rerun-if-changed={}", script.display());
            println!("cargo:rerun-if-changed={}", bundle.display());
            let status = Command::new("uv")
                .args(["run", "--no-project", "--with", "tflite==2.18.0", "python"])
                .arg(script)
                .args(["--architecture", "stgcn"])
                .arg("--bundle")
                .arg(bundle)
                .arg("--receiver")
                .arg(receiver)
                .arg("--rust-output")
                .arg(PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("stgcn_bundle.rs"))
                .status()
                .expect("run STGCN bundle validator");
            assert!(status.success(), "STGCN bundle validation failed");
            println!("cargo:rustc-cfg=stgcn");
        }
        other => panic!("unsupported FIRMWARE_MODEL: {other}"),
    }
}
