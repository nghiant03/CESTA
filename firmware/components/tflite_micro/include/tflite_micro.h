#ifndef TFLITE_MICRO_H
#define TFLITE_MICRO_H

#include <stddef.h>

#define TFLITE_MICRO_WINDOW_SIZE 60

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tflite_micro tflite_micro_t;

tflite_micro_t *tflite_micro_create(const unsigned char *model_data, size_t model_size, size_t tensor_arena_size);
void tflite_micro_destroy(tflite_micro_t *runner);
const char *tflite_micro_last_error(const tflite_micro_t *runner);
size_t tflite_micro_input_count(const tflite_micro_t *runner);
size_t tflite_micro_output_count(const tflite_micro_t *runner);
int tflite_micro_predict(
    tflite_micro_t *runner,
    const float *input,
    size_t input_count,
    float *output,
    size_t output_count
);

#ifdef __cplusplus
}
#endif

#endif
