#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CESTA_TFLITE_INPUT_COUNT 60
#define CESTA_TFLITE_OUTPUT_COUNT 6

typedef struct cesta_tflite cesta_tflite_t;

cesta_tflite_t *cesta_tflite_create(const unsigned char *model_data, size_t model_size, size_t tensor_arena_size);
void cesta_tflite_destroy(cesta_tflite_t *classifier);
const char *cesta_tflite_last_error(const cesta_tflite_t *classifier);
int cesta_tflite_predict(cesta_tflite_t *classifier, const float *input, size_t input_count, float *output, size_t output_count);

#ifdef __cplusplus
}
#endif
