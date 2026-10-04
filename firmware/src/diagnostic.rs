//! Explicit synthetic-input hardware diagnostic.

use std::thread;
use std::time::Duration;

use crate::config;
use crate::inference::{NodeClassifier, NodePass};

pub fn run() -> ! {
    log::warn!(
        "[SYNTHETIC] Diagnostic mode: fake windows and loopback payloads; radio and sensor disabled"
    );
    for remaining in (1..=15).rev() {
        log::info!(
            "[SYNTHETIC] Booted; inference starts in {} seconds",
            remaining
        );
        thread::sleep(Duration::from_secs(1));
    }
    let result = run_cases();
    loop {
        match &result {
            Ok(()) => log::info!(
                "[SYNTHETIC] PASS: all request and aggregate passes returned valid probabilities"
            ),
            Err(error) => log::error!("[SYNTHETIC] FAIL: {}", error),
        }
        thread::sleep(Duration::from_secs(5));
    }
}

fn run_cases() -> Result<(), String> {
    let senders: Vec<usize> = config::NEIGHBORS
        .iter()
        .map(|peer| peer.node_index)
        .collect();
    log::info!(
        "[SYNTHETIC] Initializing arena_bytes={}",
        config::INFERENCE_TENSOR_ARENA_BYTES
    );
    let mut classifier = NodeClassifier::new(
        config::INFERENCE_TENSOR_ARENA_BYTES,
        config::NODE_INDEX,
        &senders,
    )?;
    log::info!(
        "[SYNTHETIC] Model initialized receiver={} neighbors={} hidden={}",
        classifier.receiver_index(),
        classifier.neighbor_count(),
        classifier.hidden_size()
    );
    for scenario in ["normal", "spike", "drift", "stuck"] {
        for timestep in 0..classifier.window_size() {
            let base = 22.0 + ((timestep % 20) as f32 - 10.0) * 0.02;
            let temperature = match scenario {
                "spike" if timestep >= 58 => base + 3.0,
                "drift" => base + timestep as f32 * 0.05,
                "stuck" => 22.0,
                _ => base,
            };
            classifier.push_temperature(temperature);
        }
        log::info!("[SYNTHETIC] case={} starting request pass", scenario);
        let request = classifier.predict(None)?;
        validate_pass(&request)?;
        let requested = classifier.threshold_requests(&request.request);
        let mut slots = classifier.slots();
        let missing = classifier.predict(Some(&slots))?;
        validate_pass(&missing)?;
        report(&classifier, scenario, "missing", &missing, 0);

        // Stand in for requested neighbor replies using this synthetic window's
        // encoded rows. This exercises payload aggregation, not ESP-NOW transport.
        for (neighbor, timesteps) in requested.iter().enumerate() {
            for timestep in timesteps {
                let timestep = *timestep as usize;
                let hidden = timestep * classifier.hidden_size();
                let feature = timestep * classifier.features_per_node();
                slots.fill(
                    timestep,
                    neighbor,
                    &request.hidden[hidden..hidden + classifier.hidden_size()],
                    &classifier.features()[feature..feature + classifier.features_per_node()],
                );
            }
        }
        let aggregate = classifier.predict(Some(&slots))?;
        validate_pass(&aggregate)?;
        report(
            &classifier,
            scenario,
            "loopback",
            &aggregate,
            slots.received_count(),
        );
        log::info!(
            "[SYNTHETIC] case={} request_ms={} requested={}",
            scenario,
            request.elapsed_ms,
            requested.iter().map(Vec::len).sum::<usize>()
        );
        thread::sleep(Duration::from_millis(100));
    }
    Ok(())
}

fn validate_pass(pass: &NodePass) -> Result<(), String> {
    for probabilities in pass.probabilities.chunks_exact(4) {
        if probabilities
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
            || (probabilities.iter().sum::<f32>() - 1.0).abs() > 0.001
        {
            return Err("invalid per-timestep class probabilities".to_owned());
        }
    }
    if pass.hidden.iter().any(|value| !value.is_finite())
        || pass
            .request
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err("invalid hidden state or request probability".to_owned());
    }
    Ok(())
}

fn report(
    classifier: &NodeClassifier,
    scenario: &str,
    context: &str,
    pass: &NodePass,
    received: usize,
) {
    let diagnosis = classifier.diagnosis(pass);
    log::info!(
        "[SYNTHETIC] case={} context={} label={} probabilities={:?} elapsed_ms={} received={}",
        scenario,
        context,
        diagnosis.label,
        diagnosis.probabilities,
        pass.elapsed_ms,
        received
    );
}
