# Wake-up Word Detection Service with JNI

This project implements a wake-up word detection service for Android using JNI (Java Native Interface) and ONNX Runtime. It allows you to detect custom wake words in real-time using audio from the device's microphone.

## Features

- Real-time wake word detection using ONNX models
- Support for multiple wake word models simultaneously
- JNI interface for efficient audio processing in C++
- Android Java API for easy integration into apps

## Setup Instructions

### 1. ONNX Model Requirements

The service requires three types of ONNX models:

1. **Mel Spectrogram Model**: Converts raw audio to mel spectrograms
   - Default name: `melspectrogram.onnx`
   - Place in: `app/src/main/assets/`

2. **Embedding Model**: Converts mel spectrograms to speech embeddings
   - Default name: `embedding_model.onnx`
   - Place in: `app/src/main/assets/`

3. **Wake Word Model(s)**: Detects wake words from embeddings
   - Example: `hey_computer.onnx`
   - Place in: `app/src/main/assets/`
   - You can use multiple wake word models simultaneously

### 2. ONNX Runtime Setup

For the JNI code to work, you need to:

1. Download the ONNX Runtime Mobile Android library
2. Extract the AAR file
3. Create the following directory structure:
   ```
   app/src/main/cpp/include/        # For ONNX Runtime headers
   app/src/main/cpp/libs/arm64-v8a/ # For ARM64 ONNX Runtime library
   app/src/main/cpp/libs/armeabi-v7a/ # For ARM32 ONNX Runtime library
   app/src/main/cpp/libs/x86/      # For x86 ONNX Runtime library
   app/src/main/cpp/libs/x86_64/   # For x86_64 ONNX Runtime library
   ```

### 3. Permissions

The app requires the `RECORD_AUDIO` permission. It's already declared in the manifest, but you need to handle runtime permissions for Android 6.0+ (as done in `MainActivity`).

## Usage

### Basic Usage

```kotlin
// Create the service
val wakeupDetectorService = WakeupDetectorService(context)

// Initialize with models from assets
val melModelName = "melspectrogram.onnx"
val embModelName = "embedding_model.onnx"
val wakeWordModels = arrayOf("hey_computer.onnx")

// Add a callback
wakeupDetectorService.addCallback(object : WakeupDetectorCallback {
    override fun onWakeWordDetected(wakeWord: String) {
        // Handle wake word detection
        Log.i("WakeWordDetection", "Detected: $wakeWord")
    }
})

// Initialize and start
wakeupDetectorService.initialize(melModelName, embModelName, wakeWordModels)
wakeupDetectorService.start()

// Later, stop detection
wakeupDetectorService.stop()

// Release resources when done
wakeupDetectorService.release()
```

## Customization

You can adjust detection parameters in the native code (wakeup_detector.cpp):

- `threshold`: Activation threshold (0-1, default: 0.5)
- `triggerLevel`: Number of activations before triggering (default: 4)
- `refractory`: Frames to wait after activation (default: 20)

## Building Custom Wake Word Models

This project supports ONNX models created with the openWakeWord framework. To create custom wake word models:

1. Use the [openWakeWord](https://github.com/dscripka/openWakeWord) project to train models
2. Convert your models to ONNX format
3. Place them in the assets directory
4. Update the application code to use your model names

## Performance Considerations

- The detection runs across multiple threads for efficiency
- Audio processing is done in native code for performance
- ONNX Runtime is configured with minimal thread usage to avoid battery drain