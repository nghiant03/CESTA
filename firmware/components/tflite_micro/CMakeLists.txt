idf_component_register(
    SRCS "tflite_micro.cpp"
    INCLUDE_DIRS "include"
    REQUIRES esp-tflite-micro
)

target_compile_features(${COMPONENT_LIB} PRIVATE cxx_std_14)
