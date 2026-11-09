#include "tflite_micro.h"

#include <cstdint>
#include <cstring>
#include <new>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"
#include "tensorflow/lite/schema/schema_generated.h"

class CooperativeScheduler : public tflite::MicroProfilerInterface {
public:
    uint32_t BeginEvent(const char *) override { return 0; }
    void EndEvent(uint32_t) override {
        const TickType_t now = xTaskGetTickCount();
        if (now - last_yield_ >= pdMS_TO_TICKS(100)) {
            vTaskDelay(1);
            last_yield_ = xTaskGetTickCount();
        }
    }

private:
    TickType_t last_yield_ = 0;
};

struct tflite_micro {
    unsigned char *model_data;
    const tflite::Model *model;
    tflite::MicroMutableOpResolver<35> resolver;
    tflite::MicroInterpreter *interpreter;
    CooperativeScheduler scheduler;
    uint8_t *tensor_arena;
    const char *error;
    size_t input_count;
    size_t output_count;
};

namespace {

bool register_ops(tflite_micro_t *runner) {
    return runner->resolver.AddAbs() == kTfLiteOk &&
           runner->resolver.AddAdd() == kTfLiteOk &&
           runner->resolver.AddBatchMatMul() == kTfLiteOk &&
           runner->resolver.AddBroadcastTo() == kTfLiteOk &&
           runner->resolver.AddCast() == kTfLiteOk &&
           runner->resolver.AddConcatenation() == kTfLiteOk &&
           runner->resolver.AddDiv() == kTfLiteOk &&
           runner->resolver.AddExp() == kTfLiteOk &&
           runner->resolver.AddFullyConnected() == kTfLiteOk &&
           runner->resolver.AddGather() == kTfLiteOk &&
           runner->resolver.AddGreaterEqual() == kTfLiteOk &&
           runner->resolver.AddLog() == kTfLiteOk &&
           runner->resolver.AddLogistic() == kTfLiteOk &&
           runner->resolver.AddMaximum() == kTfLiteOk &&
           runner->resolver.AddMinimum() == kTfLiteOk &&
           runner->resolver.AddMean() == kTfLiteOk &&
           runner->resolver.AddRsqrt() == kTfLiteOk &&
           runner->resolver.AddSquaredDifference() == kTfLiteOk &&
           runner->resolver.AddMul() == kTfLiteOk &&
           runner->resolver.AddNotEqual() == kTfLiteOk &&
           runner->resolver.AddPack() == kTfLiteOk &&
           runner->resolver.AddReduceMax() == kTfLiteOk &&
           runner->resolver.AddRelu() == kTfLiteOk &&
           runner->resolver.AddReshape() == kTfLiteOk &&
           runner->resolver.AddReverseV2() == kTfLiteOk &&
           runner->resolver.AddSelectV2() == kTfLiteOk &&
           runner->resolver.AddSlice() == kTfLiteOk &&
           runner->resolver.AddSoftmax() == kTfLiteOk &&
           runner->resolver.AddSplit() == kTfLiteOk &&
           runner->resolver.AddSqrt() == kTfLiteOk &&
           runner->resolver.AddSub() == kTfLiteOk &&
           runner->resolver.AddSum() == kTfLiteOk &&
           runner->resolver.AddTanh() == kTfLiteOk &&
           runner->resolver.AddTranspose() == kTfLiteOk &&
           runner->resolver.AddUnpack() == kTfLiteOk;
}

bool has_deployment_shape(const TfLiteTensor *tensor) {
    if (tensor == nullptr || tensor->type != kTfLiteFloat32 || tensor->dims == nullptr || tensor->dims->size != 3) {
        return false;
    }
    const int width = tensor->dims->data[2];
    return tensor->dims->data[0] == 1 && tensor->dims->data[1] == TFLITE_MICRO_WINDOW_SIZE && width > 0 &&
           static_cast<size_t>(width) <= SIZE_MAX / (TFLITE_MICRO_WINDOW_SIZE * sizeof(float)) &&
           tensor->bytes == static_cast<size_t>(width) * TFLITE_MICRO_WINDOW_SIZE * sizeof(float) &&
           tensor->data.f != nullptr;
}

size_t element_count(const TfLiteTensor *tensor) {
    size_t count = 1;
    for (int index = 0; index < tensor->dims->size; ++index) {
        count *= static_cast<size_t>(tensor->dims->data[index]);
    }
    return count;
}

}

extern "C" tflite_micro_t *tflite_micro_create(const unsigned char *model_data, size_t model_size, size_t tensor_arena_size) {
    if (model_data == nullptr || model_size < 8 || tensor_arena_size == 0) {
        return nullptr;
    }

    tflite_micro_t *runner = new (std::nothrow) tflite_micro_t{};
    if (runner == nullptr) {
        return nullptr;
    }

    runner->model_data = static_cast<unsigned char *>(heap_caps_malloc(model_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (runner->model_data == nullptr) {
        runner->error = "failed to allocate model in PSRAM";
        return runner;
    }
    std::memcpy(runner->model_data, model_data, model_size);
    flatbuffers::Verifier verifier(runner->model_data, model_size);
    if (!tflite::VerifyModelBuffer(verifier)) {
        runner->error = "invalid TensorFlow Lite model buffer";
        return runner;
    }
    runner->model = tflite::GetModel(runner->model_data);
    if (runner->model == nullptr || runner->model->version() != TFLITE_SCHEMA_VERSION) {
        runner->error = "unsupported TensorFlow Lite schema";
        return runner;
    }
    if (!register_ops(runner)) {
        runner->error = "failed to register TensorFlow Lite operators";
        return runner;
    }

    runner->tensor_arena = static_cast<uint8_t *>(heap_caps_malloc(tensor_arena_size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (runner->tensor_arena == nullptr) {
        runner->error = "failed to allocate tensor arena in PSRAM";
        return runner;
    }

    runner->interpreter = new (std::nothrow)
        tflite::MicroInterpreter(runner->model, runner->resolver, runner->tensor_arena, tensor_arena_size,
                                 nullptr, &runner->scheduler);
    if (runner->interpreter == nullptr) {
        runner->error = "failed to create TensorFlow Lite interpreter";
        return runner;
    }

    const UBaseType_t priority = uxTaskPriorityGet(nullptr);
    vTaskPrioritySet(nullptr, tskIDLE_PRIORITY);
    const TfLiteStatus allocation_status = runner->interpreter->AllocateTensors();
    vTaskPrioritySet(nullptr, priority);
    if (allocation_status != kTfLiteOk) {
        runner->error = "failed to allocate TensorFlow Lite tensors";
        return runner;
    }

    if (runner->interpreter->inputs_size() != 1 || runner->interpreter->outputs_size() != 1) {
        runner->error = "stage requires exactly one input and output";
        return runner;
    }
    const TfLiteTensor *input = runner->interpreter->input(0);
    const TfLiteTensor *output = runner->interpreter->output(0);
    if (!has_deployment_shape(input) || !has_deployment_shape(output)) {
        runner->error = "model tensor shape or type does not match firmware";
        return runner;
    }
    runner->input_count = element_count(input);
    runner->output_count = element_count(output);
    ESP_LOGI("tflite_micro", "Tensor arena used: %u / %u bytes",
             static_cast<unsigned>(runner->interpreter->arena_used_bytes()), static_cast<unsigned>(tensor_arena_size));

    return runner;
}

extern "C" void tflite_micro_destroy(tflite_micro_t *runner) {
    if (runner == nullptr) {
        return;
    }
    delete runner->interpreter;
    heap_caps_free(runner->tensor_arena);
    heap_caps_free(runner->model_data);
    delete runner;
}

extern "C" const char *tflite_micro_last_error(const tflite_micro_t *runner) {
    return runner == nullptr ? "failed to allocate runner" : runner->error;
}

extern "C" int tflite_micro_predict(
    tflite_micro_t *runner,
    const float *input,
    size_t input_count,
    float *output,
    size_t output_count
) {
    if (runner == nullptr || runner->error != nullptr || runner->interpreter == nullptr ||
        input == nullptr || output == nullptr || input_count != runner->input_count ||
        output_count != runner->output_count) {
        return -1;
    }

    std::memcpy(runner->interpreter->input(0)->data.f, input, input_count * sizeof(float));
    if (runner->interpreter->Invoke() != kTfLiteOk) {
        runner->error = "TensorFlow Lite inference failed";
        return -1;
    }
    std::memcpy(output, runner->interpreter->output(0)->data.f, output_count * sizeof(float));
    return 0;
}

extern "C" size_t tflite_micro_input_count(const tflite_micro_t *runner) {
    return runner == nullptr || runner->error != nullptr ? 0 : runner->input_count;
}

extern "C" size_t tflite_micro_output_count(const tflite_micro_t *runner) {
    return runner == nullptr || runner->error != nullptr ? 0 : runner->output_count;
}
