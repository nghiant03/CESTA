use std::ffi::{CStr, c_char, c_int, c_uchar, c_void};
use std::ptr::NonNull;
use std::time::Instant;

const INPUT_COUNT: usize = 60;
const OUTPUT_COUNT: usize = 6;

static MODEL_DATA: &[u8] = include_bytes!("../model/model.tflite");

unsafe extern "C" {
    fn cesta_tflite_create(
        model_data: *const c_uchar,
        model_size: usize,
        tensor_arena_size: usize,
    ) -> *mut c_void;
    fn cesta_tflite_destroy(classifier: *mut c_void);
    fn cesta_tflite_last_error(classifier: *const c_void) -> *const c_char;
    fn cesta_tflite_predict(
        classifier: *mut c_void,
        input: *const f32,
        input_count: usize,
        output: *mut f32,
        output_count: usize,
    ) -> c_int;
}

pub struct Classifier {
    inner: NonNull<c_void>,
    samples: [f32; INPUT_COUNT],
    sample_count: usize,
}

impl Classifier {
    pub fn new(tensor_arena_size: usize) -> Result<Self, String> {
        let inner = unsafe {
            cesta_tflite_create(MODEL_DATA.as_ptr(), MODEL_DATA.len(), tensor_arena_size)
        };
        let inner = NonNull::new(inner)
            .ok_or_else(|| "failed to allocate TensorFlow Lite classifier".to_owned())?;
        let error = unsafe { cesta_tflite_last_error(inner.as_ptr()) };
        if !error.is_null() {
            let message = unsafe { CStr::from_ptr(error) }
                .to_string_lossy()
                .into_owned();
            unsafe { cesta_tflite_destroy(inner.as_ptr()) };
            return Err(message);
        }
        Ok(Self {
            inner,
            samples: [0.0; INPUT_COUNT],
            sample_count: 0,
        })
    }

    pub fn push_temperature(&mut self, temperature: f32) -> Option<Result<Inference, String>> {
        if self.sample_count < INPUT_COUNT {
            self.samples[self.sample_count] = temperature;
            self.sample_count += 1;
            if self.sample_count < INPUT_COUNT {
                return None;
            }
        } else {
            self.samples.copy_within(1.., 0);
            self.samples[INPUT_COUNT - 1] = temperature;
        }
        Some(self.predict())
    }

    fn predict(&mut self) -> Result<Inference, String> {
        let mut scores = [0.0; OUTPUT_COUNT];
        let started = Instant::now();
        let status = unsafe {
            cesta_tflite_predict(
                self.inner.as_ptr(),
                self.samples.as_ptr(),
                self.samples.len(),
                scores.as_mut_ptr(),
                scores.len(),
            )
        };
        if status != 0 {
            let error = unsafe { cesta_tflite_last_error(self.inner.as_ptr()) };
            let message = if error.is_null() {
                "TensorFlow Lite inference failed".to_owned()
            } else {
                unsafe { CStr::from_ptr(error) }
                    .to_string_lossy()
                    .into_owned()
            };
            return Err(message);
        }
        let mut class = 0;
        for index in 1..scores.len() {
            if scores[index] > scores[class] {
                class = index;
            }
        }
        Ok(Inference {
            class,
            confidence: scores[class],
            scores,
            elapsed_ms: started.elapsed().as_millis(),
        })
    }
}

impl Drop for Classifier {
    fn drop(&mut self) {
        unsafe { cesta_tflite_destroy(self.inner.as_ptr()) };
    }
}

pub struct Inference {
    pub class: usize,
    pub confidence: f32,
    pub scores: [f32; OUTPUT_COUNT],
    pub elapsed_ms: u128,
}
