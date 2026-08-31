#include "cesta_tflite.h"

#include <cstdint>
#include <cstring>
#include <new>

#include "esp_heap_caps.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

struct cesta_tflite {
    const tflite::Model *model;
    tflite::MicroMutableOpResolver<32> resolver;
    tflite::MicroInterpreter *interpreter;
    uint8_t *tensor_arena;
    const char *error;
    size_t input_count;
    size_t output_count;
};

namespace {

bool register_ops(cesta_tflite_t *classifier) {
    return classifier->resolver.AddAbs() == kTfLiteOk &&
           classifier->resolver.AddAdd() == kTfLiteOk &&
           classifier->resolver.AddBatchMatMul() == kTfLiteOk &&
           classifier->resolver.AddBroadcastTo() == kTfLiteOk &&
           classifier->resolver.AddCast() == kTfLiteOk &&
           classifier->resolver.AddConcatenation() == kTfLiteOk &&
           classifier->resolver.AddDiv() == kTfLiteOk &&
           classifier->resolver.AddExp() == kTfLiteOk &&
           classifier->resolver.AddFullyConnected() == kTfLiteOk &&
           classifier->resolver.AddGather() == kTfLiteOk &&
           classifier->resolver.AddGreaterEqual() == kTfLiteOk &&
           classifier->resolver.AddLog() == kTfLiteOk &&
           classifier->resolver.AddLogistic() == kTfLiteOk &&
           classifier->resolver.AddMaximum() == kTfLiteOk &&
           classifier->resolver.AddMinimum() == kTfLiteOk &&
           classifier->resolver.AddMul() == kTfLiteOk &&
           classifier->resolver.AddNotEqual() == kTfLiteOk &&
           classifier->resolver.AddPack() == kTfLiteOk &&
           classifier->resolver.AddReduceMax() == kTfLiteOk &&
           classifier->resolver.AddRelu() == kTfLiteOk &&
           classifier->resolver.AddReshape() == kTfLiteOk &&
           classifier->resolver.AddReverseV2() == kTfLiteOk &&
           classifier->resolver.AddSelectV2() == kTfLiteOk &&
           classifier->resolver.AddSlice() == kTfLiteOk &&
           classifier->resolver.AddSoftmax() == kTfLiteOk &&
           classifier->resolver.AddSplit() == kTfLiteOk &&
           classifier->resolver.AddSqrt() == kTfLiteOk &&
           classifier->resolver.AddSub() == kTfLiteOk &&
           classifier->resolver.AddSum() == kTfLiteOk &&
           classifier->resolver.AddTanh() == kTfLiteOk &&
           classifier->resolver.AddTranspose() == kTfLiteOk &&
           classifier->resolver.AddUnpack() == kTfLiteOk;
}

bool has_deployment_shape(const TfLiteTensor *tensor) {
    if (tensor == nullptr || tensor->type != kTfLiteFloat32 || tensor->dims == nullptr || tensor->dims->size != 3) {
        return false;
    }
    return tensor->dims->data[0] == 1 && tensor->dims->data[1] == CESTA_TFLITE_WINDOW_SIZE && tensor->dims->data[2] > 0;
}

size_t element_count(const TfLiteTensor *tensor) {
    size_t count = 1;
    for (int index = 0; index < tensor->dims->size; ++index) {
        count *= static_cast<size_t>(tensor->dims->data[index]);
    }
    return count;
}

}

extern "C" cesta_tflite_t *cesta_tflite_create(const unsigned char *model_data, size_t model_size, size_t tensor_arena_size) {
    if (model_data == nullptr || model_size < 8 || tensor_arena_size == 0) {
        return nullptr;
    }

    cesta_tflite_t *classifier = new (std::nothrow) cesta_tflite_t{};
    if (classifier == nullptr) {
        return nullptr;
    }

    classifier->model = tflite::GetModel(model_data);
    if (classifier->model == nullptr || classifier->model->version() != TFLITE_SCHEMA_VERSION) {
        classifier->error = "unsupported TensorFlow Lite schema";
        return classifier;
    }
    if (!register_ops(classifier)) {
        classifier->error = "failed to register TensorFlow Lite operators";
        return classifier;
    }

    classifier->tensor_arena = static_cast<uint8_t *>(heap_caps_malloc(tensor_arena_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (classifier->tensor_arena == nullptr) {
        classifier->error = "failed to allocate tensor arena in PSRAM";
        return classifier;
    }

    classifier->interpreter = new (std::nothrow)
        tflite::MicroInterpreter(classifier->model, classifier->resolver, classifier->tensor_arena, tensor_arena_size);
    if (classifier->interpreter == nullptr) {
        classifier->error = "failed to create TensorFlow Lite interpreter";
        return classifier;
    }
    if (classifier->interpreter->AllocateTensors() != kTfLiteOk) {
        classifier->error = "failed to allocate TensorFlow Lite tensors";
        return classifier;
    }

    const TfLiteTensor *input = classifier->interpreter->input(0);
    const TfLiteTensor *output = classifier->interpreter->output(0);
    if (!has_deployment_shape(input) || !has_deployment_shape(output)) {
        classifier->error = "model tensor shape or type does not match firmware";
        return classifier;
    }
    classifier->input_count = element_count(input);
    classifier->output_count = element_count(output);

    return classifier;
}

extern "C" void cesta_tflite_destroy(cesta_tflite_t *classifier) {
    if (classifier == nullptr) {
        return;
    }
    delete classifier->interpreter;
    heap_caps_free(classifier->tensor_arena);
    delete classifier;
}

extern "C" const char *cesta_tflite_last_error(const cesta_tflite_t *classifier) {
    return classifier == nullptr ? "failed to allocate classifier" : classifier->error;
}

extern "C" int cesta_tflite_predict(
    cesta_tflite_t *classifier,
    const float *input,
    size_t input_count,
    float *output,
    size_t output_count
) {
    if (classifier == nullptr || classifier->error != nullptr || classifier->interpreter == nullptr ||
        input == nullptr || output == nullptr || input_count != classifier->input_count ||
        output_count != classifier->output_count) {
        return -1;
    }

    std::memcpy(classifier->interpreter->input(0)->data.f, input, input_count * sizeof(float));
    if (classifier->interpreter->Invoke() != kTfLiteOk) {
        classifier->error = "TensorFlow Lite inference failed";
        return -1;
    }
    std::memcpy(output, classifier->interpreter->output(0)->data.f, output_count * sizeof(float));
    return 0;
}
