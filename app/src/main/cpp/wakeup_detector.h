#pragma once

#include <jni.h>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <onnxruntime_cxx_api.h>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <deque>

// Timestamp class for storing VAD segments
class timestamp_t {
public:
    int start;
    int end;

    timestamp_t(int start = -1, int end = -1)
        : start(start), end(end) { }

    timestamp_t& operator=(const timestamp_t& a) {
        start = a.start;
        end = a.end;
        return *this;
    }

    bool operator==(const timestamp_t& a) const {
        return (start == a.start && end == a.end);
    }
};

// VadIterator class for voice activity detection
class VadIterator {
private:
    // ONNX Runtime resources
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::shared_ptr<Ort::Session> session = nullptr;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeCPU);

    // Context-related additions
    const int context_samples = 64;  // For 16kHz, 64 samples as context
    std::vector<float> _context;     // Holds the last samples from previous chunk

    // Original window size (e.g., 32ms corresponds to 512 samples)
    int window_size_samples;
    // Effective window size = window_size_samples + context_samples
    int effective_window_size;

    // Samples per millisecond
    int sr_per_ms;

    // ONNX Runtime input/output buffers
    std::vector<Ort::Value> ort_inputs;
    std::vector<const char*> input_node_names = { "input", "state", "sr" };
    std::vector<float> input;
    unsigned int size_state = 2 * 1 * 128;
    std::vector<float> _state;
    std::vector<int64_t> sr;
    int64_t input_node_dims[2] = {};
    const int64_t state_node_dims[3] = { 2, 1, 128 };
    const int64_t sr_node_dims[1] = { 1 };
    std::vector<Ort::Value> ort_outputs;
    std::vector<const char*> output_node_names = { "output", "stateN" };

    // Model configuration parameters
    int sample_rate;
    float threshold;
    int min_silence_samples;
    int min_silence_samples_at_max_speech;
    int min_speech_samples;
    float max_speech_samples;
    int speech_pad_samples;
    int audio_length_samples;

    // State management
    bool triggered = false;
    unsigned int temp_end = 0;
    unsigned int current_sample = 0;
    int prev_end;
    int next_start = 0;
    std::vector<timestamp_t> speeches;
    timestamp_t current_speech;

    // Callback function for VAD status updates
    std::function<void(bool)> vad_callback;

    // Loads the ONNX model
    void init_onnx_model(const std::string& model_path) {
        init_engine_threads(1, 1);
        session = std::make_shared<Ort::Session>(env, model_path.c_str(), session_options);
    }

    // Initializes threading settings
    void init_engine_threads(int inter_threads, int intra_threads) {
        session_options.SetIntraOpNumThreads(intra_threads);
        session_options.SetInterOpNumThreads(inter_threads);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    }

    // Resets internal state (_state, _context, etc.)
    void reset_states() {
        std::memset(_state.data(), 0, _state.size() * sizeof(float));
        triggered = false;
        temp_end = 0;
        current_sample = 0;
        prev_end = next_start = 0;
        speeches.clear();
        current_speech = timestamp_t();
        std::fill(_context.begin(), _context.end(), 0.0f);
    }

public:
    // Process a chunk of audio data
    bool predict(const std::vector<float>& data_chunk) {
        bool was_triggered = triggered;
        
        // Build new input: first context_samples from _context, followed by the current chunk
        std::vector<float> new_data(effective_window_size, 0.0f);
        std::copy(_context.begin(), _context.end(), new_data.begin());
        std::copy(data_chunk.begin(), data_chunk.end(), new_data.begin() + context_samples);
        input = new_data;

        // Create input tensor
        Ort::Value input_ort = Ort::Value::CreateTensor<float>(
            memory_info, input.data(), input.size(), input_node_dims, 2);
        Ort::Value state_ort = Ort::Value::CreateTensor<float>(
            memory_info, _state.data(), _state.size(), state_node_dims, 3);
        Ort::Value sr_ort = Ort::Value::CreateTensor<int64_t>(
            memory_info, sr.data(), sr.size(), sr_node_dims, 1);
        ort_inputs.clear();
        ort_inputs.emplace_back(std::move(input_ort));
        ort_inputs.emplace_back(std::move(state_ort));
        ort_inputs.emplace_back(std::move(sr_ort));

        // Run inference
        ort_outputs = session->Run(
            Ort::RunOptions{ nullptr },
            input_node_names.data(), ort_inputs.data(), ort_inputs.size(),
            output_node_names.data(), output_node_names.size());

        float speech_prob = ort_outputs[0].GetTensorMutableData<float>()[0];
        float* stateN = ort_outputs[1].GetTensorMutableData<float>();
        std::memcpy(_state.data(), stateN, size_state * sizeof(float));
        current_sample += static_cast<unsigned int>(window_size_samples);

        // If speech is detected (probability >= threshold)
        if (speech_prob >= threshold) {
            if (temp_end != 0) {
                temp_end = 0;
                if (next_start < prev_end)
                    next_start = current_sample - window_size_samples;
            }
            if (!triggered) {
                triggered = true;
                current_speech.start = current_sample - window_size_samples;
                // Notify when voice activity starts
                if (vad_callback) {
                    vad_callback(true);
                }
            }
            // Update context: copy the last context_samples from new_data
            std::copy(new_data.end() - context_samples, new_data.end(), _context.begin());
            return triggered;
        }

        // If the speech segment becomes too long
        if (triggered && ((current_sample - current_speech.start) > max_speech_samples)) {
            if (prev_end > 0) {
                current_speech.end = prev_end;
                speeches.push_back(current_speech);
                current_speech = timestamp_t();
                if (next_start < prev_end)
                    triggered = false;
                else
                    current_speech.start = next_start;
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
            }
            else {
                current_speech.end = current_sample;
                speeches.push_back(current_speech);
                current_speech = timestamp_t();
                prev_end = 0;
                next_start = 0;
                temp_end = 0;
                triggered = false;
            }
            
            // Notify when voice activity ends
            if (was_triggered && !triggered && vad_callback) {
                vad_callback(false);
            }
            
            std::copy(new_data.end() - context_samples, new_data.end(), _context.begin());
            return triggered;
        }

        if ((speech_prob >= (threshold - 0.15)) && (speech_prob < threshold)) {
            // When probability drops but is still in speech, update context without changing state
            std::copy(new_data.end() - context_samples, new_data.end(), _context.begin());
            return triggered;
        }

        if (speech_prob < (threshold - 0.15)) {
            if (triggered) {
                if (temp_end == 0)
                    temp_end = current_sample;
                if (current_sample - temp_end > min_silence_samples_at_max_speech)
                    prev_end = temp_end;
                if ((current_sample - temp_end) >= min_silence_samples) {
                    current_speech.end = temp_end;
                    if (current_speech.end - current_speech.start > min_speech_samples) {
                        speeches.push_back(current_speech);
                        current_speech = timestamp_t();
                        prev_end = 0;
                        next_start = 0;
                        temp_end = 0;
                        triggered = false;
                        
                        // Notify when voice activity ends
                        if (vad_callback) {
                            vad_callback(false);
                        }
                    }
                }
            }
            std::copy(new_data.end() - context_samples, new_data.end(), _context.begin());
            return triggered;
        }
        
        return triggered;
    }

    // Process the entire audio input (will be called in chunks in real-time)
    void process(const std::vector<float>& input_wav) {
        reset_states();
        audio_length_samples = static_cast<int>(input_wav.size());
        for (size_t j = 0; j < static_cast<size_t>(audio_length_samples); j += static_cast<size_t>(window_size_samples)) {
            if (j + static_cast<size_t>(window_size_samples) > static_cast<size_t>(audio_length_samples))
                break;
            std::vector<float> chunk(&input_wav[j], &input_wav[j] + window_size_samples);
            predict(chunk);
        }
        // Handle any remaining speech segment
        if (current_speech.start >= 0 && triggered) {
            current_speech.end = audio_length_samples;
            speeches.push_back(current_speech);
            current_speech = timestamp_t();
            prev_end = 0;
            next_start = 0;
            temp_end = 0;
            triggered = false;
            
            // Notify when voice activity ends at the end of processing
            if (vad_callback) {
                vad_callback(false);
            }
        }
    }

    // Returns the detected speech timestamps
    const std::vector<timestamp_t> get_speech_timestamps() const {
        return speeches;
    }

    // Public method to reset the internal state
    void reset() {
        reset_states();
    }
    
    // Set callback for VAD status changes
    void set_callback(std::function<void(bool)> callback) {
        vad_callback = std::move(callback);
    }

    // Constructor
    VadIterator(const std::string& modelPath,
        int sample_rate = 16000, int windows_frame_size = 32,
        float threshold = 0.5, int min_silence_duration_ms = 100,
        int speech_pad_ms = 30, int min_speech_duration_ms = 250,
        float max_speech_duration_s = 30.0f)
        : sample_rate(sample_rate), threshold(threshold), 
        speech_pad_samples(speech_pad_ms * sample_rate / 1000), prev_end(0)
    {
        sr_per_ms = sample_rate / 1000;  // e.g., 16000 / 1000 = 16
        window_size_samples = windows_frame_size * sr_per_ms; // e.g., 32ms * 16 = 512 samples
        effective_window_size = window_size_samples + context_samples; // e.g., 512 + 64 = 576 samples
        input_node_dims[0] = 1;
        input_node_dims[1] = effective_window_size;
        _state.resize(size_state);
        sr.resize(1);
        sr[0] = sample_rate;
        _context.assign(context_samples, 0.0f);
        min_speech_samples = sr_per_ms * min_speech_duration_ms;
        max_speech_samples = (sample_rate * max_speech_duration_s);
        min_silence_samples = sr_per_ms * min_silence_duration_ms;
        min_silence_samples_at_max_speech = sr_per_ms * 98;
        
        reset_states();
        env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "VADDetector");
        init_onnx_model(modelPath);
    }
};

// Forward declaration for ONNX Runtime
namespace Ort {
    class Env;
    class Session;
    class Value;
}

class WakeupDetector {
public:
    WakeupDetector();
    ~WakeupDetector();

    // Initialize the detector with model paths
    bool initialize(const std::string& melModelPath, const std::string& embModelPath, 
                    const std::vector<std::string>& wakeWordModelPaths);
                    
    // Initialize VAD with model path
    bool initializeVAD(const std::string& vadModelPath);
    
    // Start listening for audio
    bool start(std::function<void(const std::string&)> wakeWordCallback);
    
    // Stop listening
    void stop();
    
    // Process audio data
    void processAudio(const int16_t* audioData, size_t numSamples);

private:
    // Constants
    static constexpr size_t chunkSamples = 1280; // 80 ms
    static constexpr size_t numMels = 32;
    static constexpr size_t embWindowSize = 76; // 775 ms
    static constexpr size_t embStepSize = 8;    // 80 ms
    static constexpr size_t embFeatures = 96;
    static constexpr size_t wwFeatures = 16;
    
    // VAD constants
    static constexpr size_t vadWindowSize = 1536; // Silero VAD window size
    static constexpr size_t vadSampleRate = 16000; // 16kHz
    
    // Audio capture constants
    static constexpr size_t audioCaptureBufferSize = vadSampleRate * 60; // 60 seconds max
    static constexpr int defaultPostSilenceMs = 500; // 0.5 seconds after voice ends
    
    // Settings
    std::string melModelPath;
    std::string embModelPath;
    std::vector<std::string> wwModelPaths;
    std::string vadModelPath;
    float threshold = 0.5f;
    int triggerLevel = 1;
    int refractory = 20;
    size_t frameSize = 4 * chunkSamples;
    size_t stepFrames = 4;
    
    // VAD settings
    std::atomic<bool> vadEnabled{false};
    std::atomic<bool> vadInitialized{false};
    float vadThreshold = 0.5f;
    int silenceLimitFrames = 10; // Process for this many frames after silence
    int silenceFrameCount = 0;   // Counter for frames of silence
    
    // VAD context-related additions (based on reference implementation)
    static constexpr int vadContextSamples = 64; // For 16kHz, 64 samples as context
    int vadMinSilenceSamples = 1600; // 100ms at 16kHz
    int vadMinSpeechSamples = 4000;  // 250ms at 16kHz 
    int vadSpeechPadSamples = 480;   // 30ms at 16kHz
    float vadMaxSpeechSamples = 16000 * 30.0f; // 30 seconds max
    std::vector<float> vadContext; // Last samples from previous chunk
    bool vadTriggered = false;     // Speech detection state
    unsigned int vadTempEnd = 0;   // Temporary end marker for speech segment
    unsigned int vadCurrentSample = 0; // Current sample position
    int vadPrevEnd = 0;
    int vadNextStart = 0;
    
    // Audio Capture settings
    std::atomic<bool> audioCaptureEnabled{false};
    std::atomic<bool> isCapturingAudio{false};
    int postSilenceCaptureDurationMs = defaultPostSilenceMs;
    int postSilenceFrameCount = 0;
    bool wakeWordDetectedFlag = false;
    std::string lastDetectedWakeWord;

public:
    // Processing threads
    std::thread melThread;
    std::thread featuresThread;
    std::vector<std::thread> wwThreads;
    std::thread vadThread;
    std::thread audioCaptureThread;
    
    // Audio processing functions
    void audioToMels();
    void melsToFeatures();
    void featuresToOutput(size_t wwIdx);
    void vadProcessing();
    
    // Thread synchronization
    std::mutex mutSamples, mutMels, mutOutput;
    std::vector<std::mutex> mutFeatures;
    std::condition_variable cvSamples, cvMels;
    std::vector<std::condition_variable> cvFeatures;
    
    // VAD synchronization
    std::mutex mutVAD;
    std::condition_variable cvVAD;
    bool vadSamplesReady = false;
    
    // Audio capture synchronization
    std::mutex mutAudioCapture;
    std::condition_variable cvAudioCapture;
    std::atomic<bool> shouldStopCapture{false};
    std::atomic<bool> audioCaptureComplete{false};
    
    // Buffers for data exchange between threads
    std::vector<float> floatSamples;
    std::vector<float> mels;
    std::vector<std::vector<float>> features;
    std::vector<float> vadSamples;
    
    // Circular buffer for audio capture
    std::deque<int16_t> audioBuffer;
    std::vector<int16_t> capturedAudio;
    
    // Thread state management
    std::atomic<bool> isRunning;
    std::atomic<bool> isInitialized;
    std::atomic<bool> samplesReady;
    std::atomic<bool> melsReady;
    std::vector<std::atomic<bool>> featuresReady;
    std::atomic<bool> isVoiceDetected{false};
    std::atomic<bool> previousVoiceState{false};
    
    // ONNX Runtime objects
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> vadSession;
    
    // VAD Iterator object
    std::unique_ptr<VadIterator> vadIterator;
    
    // VAD state for ONNX model
    static constexpr unsigned int vadStateSize = 2 * 1 * 128;
    std::vector<float> vadState;
    
    // Callback when wake word is detected
    std::function<void(const std::string&)> wakeWordCallback;
    // Callback for VAD status
    std::function<void(bool)> vadCallback;
    std::string vadInputNameStr;
    std::string vadOutputNameStr;
    
    std::vector<std::string> vadInputNames;
    std::vector<std::string> vadOutputNames;
    // Memory info for VAD
    std::vector<int64_t> vadInputShape;
    // vadInputLength
    size_t vadInputLength = 0;
    // Callback for audio capture completion
    std::function<void(const std::vector<int16_t>&, int)> audioCaptureCallback;
};