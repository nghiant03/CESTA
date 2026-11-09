use std::ffi::{CStr, c_char, c_int, c_void};
use std::ptr::NonNull;

unsafe extern "C" {
    fn tflite_micro_create(model: *const u8, size: usize, arena_size: usize) -> *mut c_void;
    fn tflite_micro_destroy(runner: *mut c_void);
    fn tflite_micro_last_error(runner: *const c_void) -> *const c_char;
    fn tflite_micro_input_count(runner: *const c_void) -> usize;
    fn tflite_micro_output_count(runner: *const c_void) -> usize;
    fn tflite_micro_predict(
        runner: *mut c_void,
        input: *const f32,
        input_count: usize,
        output: *mut f32,
        output_count: usize,
    ) -> c_int;
}

pub struct StageRunner {
    inner: Option<NonNull<c_void>>,
    arena_size: usize,
}

impl StageRunner {
    pub fn new(arena_size: usize) -> Result<Self, String> {
        if arena_size == 0 {
            return Err("tensor arena size must be positive".to_owned());
        }
        Ok(Self {
            inner: None,
            arena_size,
        })
    }

    pub fn unload(&mut self) {
        if let Some(inner) = self.inner.take() {
            unsafe { tflite_micro_destroy(inner.as_ptr()) };
        }
    }

    pub fn load(
        &mut self,
        model: &[u8],
        input_width: usize,
        output_width: usize,
    ) -> Result<(), String> {
        self.unload();
        let input_count = input_width
            .checked_mul(60)
            .filter(|count| *count > 0)
            .ok_or_else(|| "invalid stage input width".to_owned())?;
        let output_count = output_width
            .checked_mul(60)
            .filter(|count| *count > 0)
            .ok_or_else(|| "invalid stage output width".to_owned())?;
        let inner = NonNull::new(unsafe {
            tflite_micro_create(model.as_ptr(), model.len(), self.arena_size)
        })
        .ok_or_else(|| "failed to allocate TensorFlow Lite runner".to_owned())?;
        self.inner = Some(inner);
        if let Some(error) = self.error() {
            self.unload();
            return Err(error);
        }
        if unsafe { tflite_micro_input_count(inner.as_ptr()) } != input_count
            || unsafe { tflite_micro_output_count(inner.as_ptr()) } != output_count
        {
            self.unload();
            return Err("stage tensor widths do not match metadata".to_owned());
        }
        Ok(())
    }

    pub fn predict(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), String> {
        let inner = self.inner.ok_or_else(|| "no stage loaded".to_owned())?;
        if input.iter().any(|value| !value.is_finite()) {
            return Err("stage input contains nonfinite values".to_owned());
        }
        let status = unsafe {
            tflite_micro_predict(
                inner.as_ptr(),
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                output.len(),
            )
        };
        if status != 0 {
            return Err(self
                .error()
                .unwrap_or_else(|| "stage tensor lengths do not match model".to_owned()));
        }
        if output.iter().any(|value| !value.is_finite()) {
            return Err("model returned nonfinite output".to_owned());
        }
        Ok(())
    }

    fn error(&self) -> Option<String> {
        let inner = self.inner?;
        let error = unsafe { tflite_micro_last_error(inner.as_ptr()) };
        if error.is_null() {
            None
        } else {
            Some(
                unsafe { CStr::from_ptr(error) }
                    .to_string_lossy()
                    .into_owned(),
            )
        }
    }
}

impl Drop for StageRunner {
    fn drop(&mut self) {
        self.unload();
    }
}
