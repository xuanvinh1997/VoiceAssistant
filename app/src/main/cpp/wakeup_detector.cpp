#include "wakeup_detector.h"
#include <android/log.h>
#include <onnxruntime_cxx_api.h>
#include <filesystem>
#include <numeric>
#include <string>

#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "WakeupDetector", __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "WakeupDetector", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "WakeupDetector", __VA_ARGS__)

// Global callback to pass data from C++ to Java
static JavaVM* javaVM = nullptr;
static jweak wakeupDetectorCallback = nullptr;
static jmethodID onWakeWordDetectedMethod = nullptr;
static jmethodID onDetectionScoreUpdateMethod = nullptr;

// JNI helper function to handle wake word detection callback
void notifyWakeWordDetected(const std::string& wakeWord) {
    JNIEnv* env;
    bool detach = false;
    int getEnvStat = javaVM->GetEnv((void**)&env, JNI_VERSION_1_6);
    
    if (getEnvStat == JNI_EDETACHED) {
        javaVM->AttachCurrentThread(&env, nullptr);
        detach = true;
    }
    
    if (env && wakeupDetectorCallback) {
        jobject callback = env->NewLocalRef(wakeupDetectorCallback);
        if (callback) {
            jstring jWakeWord = env->NewStringUTF(wakeWord.c_str());
            env->CallVoidMethod(callback, onWakeWordDetectedMethod, jWakeWord);
            env->DeleteLocalRef(jWakeWord);
            env->DeleteLocalRef(callback);
            
            if (env->ExceptionCheck()) {
                env->ExceptionDescribe();
                env->ExceptionClear();
            }
        }
    }
    
    if (detach) {
        javaVM->DetachCurrentThread();
    }
}

// JNI helper function to handle detection score updates
void notifyDetectionScoreUpdate(const std::string& wakeWord, float score, float threshold, 
                               int activation, int triggerLevel) {
    JNIEnv* env;
    bool detach = false;
    int getEnvStat = javaVM->GetEnv((void**)&env, JNI_VERSION_1_6);
    
    if (getEnvStat == JNI_EDETACHED) {
        javaVM->AttachCurrentThread(&env, nullptr);
        detach = true;
    }
    
    if (env && wakeupDetectorCallback && onDetectionScoreUpdateMethod) {
        jobject callback = env->NewLocalRef(wakeupDetectorCallback);
        if (callback) {
            jstring jWakeWord = env->NewStringUTF(wakeWord.c_str());
            env->CallVoidMethod(callback, onDetectionScoreUpdateMethod, jWakeWord, 
                               score, threshold, activation, triggerLevel);
            env->DeleteLocalRef(jWakeWord);
            env->DeleteLocalRef(callback);
            
            if (env->ExceptionCheck()) {
                env->ExceptionDescribe();
                env->ExceptionClear();
            }
        }
    }
    
    if (detach) {
        javaVM->DetachCurrentThread();
    }
}

// Constructor
WakeupDetector::WakeupDetector() 
    : isRunning(false), isInitialized(false), samplesReady(false), melsReady(false) {
    LOGI("WakeupDetector constructor called");
}

// Destructor
WakeupDetector::~WakeupDetector() {
    LOGI("WakeupDetector destructor called");
    stop();
}

// Initialize with model paths
bool WakeupDetector::initialize(const std::string& melModelPath, const std::string& embModelPath, 
                               const std::vector<std::string>& wakeWordModelPaths) {
    LOGI("Initializing WakeupDetector with models");
    
    this->melModelPath = melModelPath;
    this->embModelPath = embModelPath;
    this->wwModelPaths = wakeWordModelPaths;
    
    if (wwModelPaths.empty()) {
        LOGE("No wake word models provided");
        return false;
    }
    
    try {
        // Initialize ONNX Runtime
        env = std::make_unique<Ort::Env>(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "WakeupDetector");
        env->DisableTelemetryEvents();
        
        // Initialize mutexes and condition variables for wake word models
        const size_t numWakeWords = wwModelPaths.size();
        // Replace resize calls with direct initialization to avoid copy/move operations
        mutFeatures = std::vector<std::mutex>(numWakeWords);
        cvFeatures = std::vector<std::condition_variable>(numWakeWords);
        
        // Initialize atomic vector with proper size (no resize)
        featuresReady = std::vector<std::atomic<bool>>(numWakeWords);
        features.resize(numWakeWords);
        
        for (auto& ready : featuresReady) {
            ready = false;
        }
        
        isInitialized = true;
        LOGI("WakeupDetector initialized successfully with %zu wake word models", numWakeWords);
        return true;
    } catch (const std::exception& e) {
        LOGE("Error initializing WakeupDetector: %s", e.what());
        return false;
    }
}

// Start detection
bool WakeupDetector::start(std::function<void(const std::string&)> callback) {
    if (!isInitialized) {
        LOGE("Cannot start detector: not initialized");
        return false;
    }
    
    if (isRunning) {
        LOGI("Detector is already running");
        return true;
    }
    
    try {
        LOGI("Starting WakeupDetector");
        
        // Set callback
        wakeWordCallback = std::move(callback);
        
        // Reset state
        isRunning = true;
        samplesReady = false;
        melsReady = false;
        
        for (auto& ready : featuresReady) {
            ready = false;
        }
        
        floatSamples.clear();
        mels.clear();
        for (auto& feature : features) {
            feature.clear();
        }
        
        // Start threads
        melThread = std::thread(&WakeupDetector::audioToMels, this);
        featuresThread = std::thread(&WakeupDetector::melsToFeatures, this);
        
        wwThreads.clear();
        for (size_t i = 0; i < wwModelPaths.size(); i++) {
            wwThreads.emplace_back(&WakeupDetector::featuresToOutput, this, i);
        }
        
        LOGI("WakeupDetector started successfully");
        return true;
    } catch (const std::exception& e) {
        LOGE("Error starting detector: %s", e.what());
        isRunning = false;
        return false;
    }
}

// Stop detection
void WakeupDetector::stop() {
    if (!isRunning) {
        return;
    }
    
    LOGI("Stopping WakeupDetector");
    
    isRunning = false;
    
    // Wake up all threads to check isRunning and exit
    {
        std::unique_lock<std::mutex> lockSamples(mutSamples);
        samplesReady = true;
        cvSamples.notify_one();
    }
    
    {
        std::unique_lock<std::mutex> lockMels(mutMels);
        melsReady = true;
        cvMels.notify_one();
    }
    
    for (size_t i = 0; i < featuresReady.size(); i++) {
        std::unique_lock<std::mutex> lockFeatures(mutFeatures[i]);
        featuresReady[i] = true;
        cvFeatures[i].notify_one();
    }
    
    // Join threads
    if (melThread.joinable()) melThread.join();
    if (featuresThread.joinable()) featuresThread.join();
    
    for (auto& thread : wwThreads) {
        if (thread.joinable()) thread.join();
    }
    
    LOGI("WakeupDetector stopped");
}

// Process audio data
void WakeupDetector::processAudio(const int16_t* audioData, size_t numSamples) {
    if (!isRunning || !audioData || numSamples == 0) {
        return;
    }
    
    std::unique_lock<std::mutex> lockSamples(mutSamples);
    
    // Convert int16_t samples to float and add to buffer
    for (size_t i = 0; i < numSamples; i++) {
        floatSamples.push_back(static_cast<float>(audioData[i]));
    }
    
    samplesReady = true;
    cvSamples.notify_one();
}

// Audio to mel spectrogram conversion thread
void WakeupDetector::audioToMels() {
    LOGI("audioToMels thread started");
    
    try {
        // Create ONNX Runtime session for mel spectrogram model
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetInterOpNumThreads(1);
        
        auto melSession = Ort::Session(*env, melModelPath.c_str(), sessionOptions);
        
        Ort::AllocatorWithDefaultOptions allocator;
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        
        auto melInputName = melSession.GetInputNameAllocated(0, allocator);
        std::vector<const char*> melInputNames{melInputName.get()};
        
        auto melOutputName = melSession.GetOutputNameAllocated(0, allocator);
        std::vector<const char*> melOutputNames{melOutputName.get()};
        
        std::vector<float> todoSamples;
        std::vector<int64_t> samplesShape{1, static_cast<int64_t>(frameSize)};
        
        LOGI("Mel spectrogram model loaded");
        
        while (isRunning) {
            {
                std::unique_lock<std::mutex> lockSamples(mutSamples);
                cvSamples.wait(lockSamples, [this] { 
                    return samplesReady || !isRunning; 
                });
                
                if (!isRunning) break;
                
                // Copy samples to processing buffer
                todoSamples.insert(todoSamples.end(), floatSamples.begin(), floatSamples.end());
                floatSamples.clear();
                samplesReady = false;
            }
            
            // Process samples in frameSize chunks
            while (todoSamples.size() >= frameSize && isRunning) {
                // Generate mels for audio samples
                std::vector<Ort::Value> melInputTensors;
                melInputTensors.push_back(Ort::Value::CreateTensor<float>(
                    memoryInfo, todoSamples.data(), frameSize, samplesShape.data(), 
                    samplesShape.size()));
                
                auto melOutputTensors = melSession.Run(Ort::RunOptions{nullptr}, 
                    melInputNames.data(), melInputTensors.data(), melInputNames.size(),
                    melOutputNames.data(), melOutputNames.size());
                
                const auto& melOut = melOutputTensors.front();
                const auto melInfo = melOut.GetTensorTypeAndShapeInfo();
                const auto melShape = melInfo.GetShape();
                
                const float* melData = melOut.GetTensorData<float>();
                size_t melCount = std::accumulate(melShape.begin(), melShape.end(), 
                                                 1, std::multiplies<>());
                
                {
                    std::unique_lock<std::mutex> lockMels(mutMels);
                    for (size_t i = 0; i < melCount; i++) {
                        // Scale mels for Google speech embedding model
                        mels.push_back((melData[i] / 10.0f) + 2.0f);
                    }
                    melsReady = true;
                    cvMels.notify_one();
                }
                
                // Remove processed samples
                todoSamples.erase(todoSamples.begin(), 
                                 todoSamples.begin() + frameSize);
            }
        }
    } catch (const std::exception& e) {
        LOGE("Error in audioToMels: %s", e.what());
    }
    
    LOGI("audioToMels thread exiting");
}

// Mel spectrogram to embedding features thread
void WakeupDetector::melsToFeatures() {
    LOGI("melsToFeatures thread started");
    
    try {
        // Create ONNX Runtime session for embedding model
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetInterOpNumThreads(1);
        
        auto embSession = Ort::Session(*env, embModelPath.c_str(), sessionOptions);
        
        Ort::AllocatorWithDefaultOptions allocator;
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        
        auto embInputName = embSession.GetInputNameAllocated(0, allocator);
        std::vector<const char*> embInputNames{embInputName.get()};
        
        auto embOutputName = embSession.GetOutputNameAllocated(0, allocator);
        std::vector<const char*> embOutputNames{embOutputName.get()};
        
        std::vector<float> todoMels;
        size_t melFrames = 0;
        std::vector<int64_t> embShape{1, static_cast<int64_t>(embWindowSize), 
                                     static_cast<int64_t>(numMels), 1};
        
        LOGI("Embedding model loaded");
        
        while (isRunning) {
            {
                std::unique_lock<std::mutex> lockMels(mutMels);
                cvMels.wait(lockMels, [this] { 
                    return melsReady || !isRunning; 
                });
                
                if (!isRunning) break;
                
                // Copy mels to processing buffer
                todoMels.insert(todoMels.end(), mels.begin(), mels.end());
                mels.clear();
                melsReady = false;
            }
            
            melFrames = todoMels.size() / numMels;
            while (melFrames >= embWindowSize && isRunning) {
                // Generate embeddings for mels
                std::vector<Ort::Value> embInputTensors;
                embInputTensors.push_back(Ort::Value::CreateTensor<float>(
                    memoryInfo, todoMels.data(), embWindowSize * numMels, 
                    embShape.data(), embShape.size()));
                
                auto embOutputTensors = embSession.Run(Ort::RunOptions{nullptr}, 
                    embInputNames.data(), embInputTensors.data(), embInputTensors.size(),
                    embOutputNames.data(), embOutputNames.size());
                
                const auto& embOut = embOutputTensors.front();
                const auto embOutInfo = embOut.GetTensorTypeAndShapeInfo();
                const auto embOutShape = embOutInfo.GetShape();
                
                const float* embOutData = embOut.GetTensorData<float>();
                size_t embOutCount = std::accumulate(embOutShape.begin(), embOutShape.end(), 
                                                    1, std::multiplies<>());
                
                // Send to each wake word model
                for (size_t i = 0; i < features.size() && isRunning; i++) {
                    std::unique_lock<std::mutex> lockFeatures(mutFeatures[i]);
                    std::copy(embOutData, embOutData + embOutCount, 
                            std::back_inserter(features[i]));
                    featuresReady[i] = true;
                    cvFeatures[i].notify_one();
                }
                
                // Erase a step's worth of mels
                todoMels.erase(todoMels.begin(),
                              todoMels.begin() + (embStepSize * numMels));
                
                melFrames = todoMels.size() / numMels;
            }
        }
    } catch (const std::exception& e) {
        LOGE("Error in melsToFeatures: %s", e.what());
    }
    
    LOGI("melsToFeatures thread exiting");
}

// Features to wake word detection thread
void WakeupDetector::featuresToOutput(size_t wwIdx) {
    if (wwIdx >= wwModelPaths.size()) {
        LOGE("Invalid wake word model index: %zu", wwIdx);
        return;
    }
    
    LOGI("featuresToOutput thread %zu started", wwIdx);
    
    try {
        // Create ONNX Runtime session for wake word model
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetInterOpNumThreads(1);
        
        auto wwModelPath = wwModelPaths[wwIdx];
        
        // Extract wake word name from filename
        std::string wwName = std::filesystem::path(wwModelPath).stem().string();
        
        auto wwSession = Ort::Session(*env, wwModelPath.c_str(), sessionOptions);
        
        Ort::AllocatorWithDefaultOptions allocator;
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        
        auto wwInputName = wwSession.GetInputNameAllocated(0, allocator);
        std::vector<const char*> wwInputNames{wwInputName.get()};
        
        auto wwOutputName = wwSession.GetOutputNameAllocated(0, allocator);
        std::vector<const char*> wwOutputNames{wwOutputName.get()};
        
        std::vector<float> todoFeatures;
        size_t numBufferedFeatures = 0;
        int activation = 0;
        std::vector<int64_t> wwShape{1, static_cast<int64_t>(wwFeatures), 
                                    static_cast<int64_t>(embFeatures)};
        
        LOGI("Wake word model %s loaded", wwName.c_str());
        
        // For logging scores
        int logCounter = 0;
        const int logFrequency = 20; // Log every 20th score to avoid flooding logs
        
        while (isRunning) {
            {
                std::unique_lock<std::mutex> lockFeatures(mutFeatures[wwIdx]);
                cvFeatures[wwIdx].wait(lockFeatures, [this, wwIdx] { 
                    return featuresReady[wwIdx] || !isRunning; 
                });
                
                if (!isRunning) break;
                
                // Copy features to processing buffer
                todoFeatures.insert(todoFeatures.end(), features[wwIdx].begin(), 
                                    features[wwIdx].end());
                features[wwIdx].clear();
                featuresReady[wwIdx] = false;
            }
            
            numBufferedFeatures = todoFeatures.size() / embFeatures;
            while (numBufferedFeatures >= wwFeatures && isRunning) {
                std::vector<Ort::Value> wwInputTensors;
                wwInputTensors.push_back(Ort::Value::CreateTensor<float>(
                    memoryInfo, todoFeatures.data(), wwFeatures * embFeatures,
                    wwShape.data(), wwShape.size()));
                
                auto wwOutputTensors = wwSession.Run(Ort::RunOptions{nullptr}, 
                    wwInputNames.data(), wwInputTensors.data(), 1, 
                    wwOutputNames.data(), 1);
                
                const auto& wwOut = wwOutputTensors.front();
                const auto wwOutInfo = wwOut.GetTensorTypeAndShapeInfo();
                const auto wwOutShape = wwOutInfo.GetShape();
                const float* wwOutData = wwOut.GetTensorData<float>();
                size_t wwOutCount = std::accumulate(wwOutShape.begin(), wwOutShape.end(), 
                                                   1, std::multiplies<>());
                
                for (size_t i = 0; i < wwOutCount; i++) {
                    auto probability = wwOutData[i];
                    
                    // Log detection score periodically or if it's close to threshold
                    logCounter++;
                    if (logCounter % logFrequency == 0 || probability > threshold * 0.7) {
                        LOGD("[%s] Detection score: %.4f (threshold: %.2f, activation: %d/%d)", 
                            wwName.c_str(), probability, threshold, activation, triggerLevel);
                    }
                    
                    if (probability > threshold) {
                        // Activated
                        activation++;
                        LOGI("[%s] Score %.4f exceeded threshold (%.2f), activation %d/%d", 
                             wwName.c_str(), probability, threshold, activation, triggerLevel);
                        
                        if (activation >= triggerLevel) {
                            // Trigger level reached
                            LOGI("Wake word detected: %s (score: %.4f)", wwName.c_str(), probability);
                            
                            if (wakeWordCallback) {
                                wakeWordCallback(wwName);
                            }
                            
                            activation = -refractory;
                        }
                    } else {
                        // Back towards 0
                        if (activation > 0) {
                            activation = std::max(0, activation - 1);
                            if (logCounter % logFrequency == 0) {
                                LOGD("[%s] Activation decaying: %d", wwName.c_str(), activation);
                            }
                        } else {
                            activation = std::min(0, activation + 1);
                        }
                    }
                }
                
                // Remove 1 embedding
                todoFeatures.erase(todoFeatures.begin(),
                                  todoFeatures.begin() + (1 * embFeatures));
                
                numBufferedFeatures = todoFeatures.size() / embFeatures;
            }
        }
    } catch (const std::exception& e) {
        LOGE("Error in featuresToOutput (%zu): %s", wwIdx, e.what());
    }
    
    LOGI("featuresToOutput thread %zu exiting", wwIdx);
}

// JNI implementation
extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved) {
    javaVM = vm;
    JNIEnv* env;
    if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
        return JNI_ERR;
    }
    
    // Get the WakeupDetectorCallback class
    jclass callbackClass = env->FindClass("com/vinhpx/voiceassistant/WakeupDetectorCallback");
    if (callbackClass == nullptr) {
        LOGE("Failed to find WakeupDetectorCallback class");
        return JNI_ERR;
    }
    
    // Get the onWakeWordDetected method ID
    onWakeWordDetectedMethod = env->GetMethodID(callbackClass, "onWakeWordDetected", "(Ljava/lang/String;)V");
    if (onWakeWordDetectedMethod == nullptr) {
        LOGE("Failed to find onWakeWordDetected method");
        return JNI_ERR;
    }
    
    // Get the onDetectionScoreUpdate method ID
    onDetectionScoreUpdateMethod = env->GetMethodID(callbackClass, "onDetectionScoreUpdate", "(Ljava/lang/String;FFII)V");
    if (onDetectionScoreUpdateMethod == nullptr) {
        LOGE("Failed to find onDetectionScoreUpdate method");
        return JNI_ERR;
    }
    
    env->DeleteLocalRef(callbackClass);
    
    return JNI_VERSION_1_6;
}

JNIEXPORT jlong JNICALL Java_com_vinhpx_voiceassistant_WakeupDetectorJNI_createWakeupDetector(
        JNIEnv* env, jobject thiz) {
    LOGI("Creating WakeupDetector");
    auto* detector = new WakeupDetector();
    return reinterpret_cast<jlong>(detector);
}

JNIEXPORT jboolean JNICALL Java_com_vinhpx_voiceassistant_WakeupDetectorJNI_initializeDetector(
        JNIEnv* env, jobject thiz, jlong detectorPtr, jstring melModelPath, 
        jstring embModelPath, jobjectArray wwModelPaths) {
    auto* detector = reinterpret_cast<WakeupDetector*>(detectorPtr);
    if (!detector) return JNI_FALSE;
    
    // Convert Java strings to C++ strings
    const char* melModelChars = env->GetStringUTFChars(melModelPath, nullptr);
    const char* embModelChars = env->GetStringUTFChars(embModelPath, nullptr);
    
    std::string melModelStr(melModelChars);
    std::string embModelStr(embModelChars);
    
    env->ReleaseStringUTFChars(melModelPath, melModelChars);
    env->ReleaseStringUTFChars(embModelPath, embModelChars);
    
    // Convert Java string array to C++ vector of strings
    std::vector<std::string> wwModelPathsVec;
    jsize wwModelCount = env->GetArrayLength(wwModelPaths);
    
    for (jsize i = 0; i < wwModelCount; i++) {
        jstring pathStr = (jstring)env->GetObjectArrayElement(wwModelPaths, i);
        const char* pathChars = env->GetStringUTFChars(pathStr, nullptr);
        wwModelPathsVec.emplace_back(pathChars);
        env->ReleaseStringUTFChars(pathStr, pathChars);
        env->DeleteLocalRef(pathStr);
    }
    
    return detector->initialize(melModelStr, embModelStr, wwModelPathsVec) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL Java_com_vinhpx_voiceassistant_WakeupDetectorJNI_startDetector(
        JNIEnv* env, jobject thiz, jlong detectorPtr) {
    auto* detector = reinterpret_cast<WakeupDetector*>(detectorPtr);
    if (!detector) return JNI_FALSE;
    
    // Store the callback object as a global reference
    if (wakeupDetectorCallback) {
        env->DeleteWeakGlobalRef(wakeupDetectorCallback);
        wakeupDetectorCallback = nullptr;
    }
    
    // Create a global reference to the callback object
    wakeupDetectorCallback = env->NewWeakGlobalRef(thiz);
    
    // Start the detector with a C++ callback that calls the Java callback
    return detector->start([](const std::string& wakeWord) {
        notifyWakeWordDetected(wakeWord);
    }) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL Java_com_vinhpx_voiceassistant_WakeupDetectorJNI_stopDetector(
        JNIEnv* env, jobject thiz, jlong detectorPtr) {
    auto* detector = reinterpret_cast<WakeupDetector*>(detectorPtr);
    if (detector) {
        detector->stop();
    }
}

JNIEXPORT void JNICALL Java_com_vinhpx_voiceassistant_WakeupDetectorJNI_processAudio(
        JNIEnv* env, jobject thiz, jlong detectorPtr, jshortArray audioData, jint numSamples) {
    auto* detector = reinterpret_cast<WakeupDetector*>(detectorPtr);
    if (!detector) return;
    
    // Get the audio data from Java
    jshort* audioBuffer = env->GetShortArrayElements(audioData, nullptr);
    if (!audioBuffer) return;
    
    // Process the audio
    detector->processAudio(reinterpret_cast<int16_t*>(audioBuffer), numSamples);
    
    // Release the Java array
    env->ReleaseShortArrayElements(audioData, audioBuffer, JNI_ABORT);
}

JNIEXPORT void JNICALL Java_com_vinhpx_voiceassistant_WakeupDetectorJNI_destroyWakeupDetector(
        JNIEnv* env, jobject thiz, jlong detectorPtr) {
    auto* detector = reinterpret_cast<WakeupDetector*>(detectorPtr);
    if (detector) {
        delete detector;
    }
    
    // Clean up global reference if needed
    if (wakeupDetectorCallback) {
        env->DeleteWeakGlobalRef(wakeupDetectorCallback);
        wakeupDetectorCallback = nullptr;
    }
}

} // extern "C"