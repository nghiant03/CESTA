pub struct MinuteBuckets {
    start: u64,
    cadence: u64,
    sum: f64,
    count: u32,
}

pub struct CompletedBucket {
    pub timestamp: u64,
    pub temperature: Option<f32>,
    pub count: u32,
}

impl MinuteBuckets {
    pub fn new(now: u64, cadence: u64) -> Self {
        assert!(cadence > 0);
        Self {
            start: (now / cadence + 1) * cadence,
            cadence,
            sum: 0.0,
            count: 0,
        }
    }

    pub fn close(&mut self, now: u64) -> Option<CompletedBucket> {
        if now < self.start || now < self.start + self.cadence {
            return None;
        }
        let timestamp = now / self.cadence * self.cadence - self.cadence;
        let current = timestamp == self.start;
        let bucket = CompletedBucket {
            timestamp,
            temperature: (current && self.count > 0).then(|| (self.sum / self.count as f64) as f32),
            count: if current { self.count } else { 0 },
        };
        self.start = timestamp + self.cadence;
        self.sum = 0.0;
        self.count = 0;
        Some(bucket)
    }

    pub fn add(&mut self, timestamp: u64, temperature: f32) {
        if timestamp >= self.start
            && timestamp < self.start + self.cadence
            && temperature.is_finite()
        {
            self.sum += temperature as f64;
            self.count += 1;
        }
    }
}

pub fn bucket_is_fresh(timestamp: u64, now: u64, cadence: u64) -> bool {
    now >= timestamp + cadence && now < timestamp + 2 * cadence
}
