cmake_minimum_required(VERSION 3.15)

project(SUNOKILLER_VST VERSION 0.1.0)

# C++17 required for modern features
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# JUCE Framework (add as submodule or download)
# git submodule add https://github.com/juce-framework/JUCE.git plugins/vst/JUCE
add_subdirectory(JUCE)

# Plugin configuration
juce_add_plugin(SUNOKILLER
    COMPANY_NAME "MASSIVEMAGNETICS"
    PLUGIN_MANUFACTURER_CODE Msmg
    PLUGIN_CODE Snkl
    FORMATS VST3 AU Standalone
    PRODUCT_NAME "SUNOKILLER"
    
    # Plugin characteristics
    IS_SYNTH TRUE
    NEEDS_MIDI_INPUT TRUE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS TRUE
    
    # Copy step
    COPY_PLUGIN_AFTER_BUILD FALSE
    
    # Metadata
    PLUGIN_MANUFACTURER "MASSIVEMAGNETICS"
    PLUGIN_DESCRIPTION "AI-powered music and voice synthesis"
    PLUGIN_CATEGORY "Instrument"
)

# Source files
target_sources(SUNOKILLER PRIVATE
    Source/PluginProcessor.cpp
    Source/PluginEditor.cpp
    Source/ModelWrapper.cpp
)

# JUCE modules required
target_compile_definitions(SUNOKILLER PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
)

target_link_libraries(SUNOKILLER PRIVATE
    juce::juce_audio_basics
    juce::juce_audio_devices
    juce::juce_audio_formats
    juce::juce_audio_plugin_client
    juce::juce_audio_processors
    juce::juce_audio_utils
    juce::juce_core
    juce::juce_data_structures
    juce::juce_events
    juce::juce_graphics
    juce::juce_gui_basics
    juce::juce_gui_extra
)

# LibTorch for model inference (optional - can use ONNX instead)
# set(CMAKE_PREFIX_PATH "path/to/libtorch")
# find_package(Torch REQUIRED)
# target_link_libraries(SUNOKILLER PRIVATE "${TORCH_LIBRARIES}")

# ONNX Runtime for model inference (recommended for plugins)
# Download from https://github.com/microsoft/onnxruntime/releases
# set(ONNXRUNTIME_ROOT "path/to/onnxruntime")
# target_include_directories(SUNOKILLER PRIVATE ${ONNXRUNTIME_ROOT}/include)
# target_link_libraries(SUNOKILLER PRIVATE ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib)

# Installation
install(TARGETS SUNOKILLER
    LIBRARY DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
    RUNTIME DESTINATION "${CMAKE_INSTALL_PREFIX}/bin"
)
