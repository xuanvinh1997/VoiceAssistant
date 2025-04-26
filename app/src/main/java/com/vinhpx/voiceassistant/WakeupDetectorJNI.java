package com.vinhpx.voiceassistant;

/**
 * JNI wrapper for the native wake-up word detector.
 */
public class WakeupDetectorJNI implements WakeupDetectorCallback {

    // Load the native library
    static {
        System.loadLibrary("voiceassistant");
    }

    // Pointer to the native WakeupDetector object
    private long nativeDetectorPtr;
    private WakeupDetectorCallback callback;

    /**
     * Create a new WakeupDetectorJNI instance
     */
    public WakeupDetectorJNI() {
        nativeDetectorPtr = createWakeupDetector();
    }

    /**
     * Initialize the detector with model paths
     *
     * @param melModelPath Path to the mel spectrogram ONNX model
     * @param embModelPath Path to the embedding ONNX model
     * @param wwModelPaths Array of paths to wake word ONNX models
     * @return true if initialization succeeded
     */
    public boolean initialize(String melModelPath, String embModelPath, String[] wwModelPaths) {
        return initializeDetector(nativeDetectorPtr, melModelPath, embModelPath, wwModelPaths);
    }
    
    /**
     * Initialize the VAD detector with model path
     * 
     * @param vadModelPath Path to the VAD (Voice Activity Detection) ONNX model
     * @return true if initialization succeeded
     */
    public boolean initializeVAD(String vadModelPath) {
        return initializeVAD(nativeDetectorPtr, vadModelPath);
    }

    /**
     * Start the detector
     *
     * @return true if starting succeeded
     */
    public boolean start() {
        return startDetector(nativeDetectorPtr);
    }

    /**
     * Stop the detector
     */
    public void stop() {
        stopDetector(nativeDetectorPtr);
    }

    /**
     * Process audio data
     *
     * @param audioData Array of audio samples (16-bit PCM)
     * @param numSamples Number of samples in the array
     */
    public void processAudio(short[] audioData, int numSamples) {
        processAudio(nativeDetectorPtr, audioData, numSamples);
    }

    /**
     * Set the callback for wake word detection events
     *
     * @param callback The callback to receive events
     */
    public void setCallback(WakeupDetectorCallback callback) {
        this.callback = callback;
    }

    /**
     * Release native resources
     */
    public void release() {
        destroyWakeupDetector(nativeDetectorPtr);
        nativeDetectorPtr = 0;
    }

    @Override
    public void onWakeWordDetected(String wakeWord) {
        if (callback != null) {
            callback.onWakeWordDetected(wakeWord);
        }
    }
    
    @Override
    public void onDetectionScoreUpdate(String wakeWord, float score, float threshold, 
                                      int activation, int triggerLevel) {
        if (callback != null) {
            callback.onDetectionScoreUpdate(wakeWord, score, threshold, activation, triggerLevel);
        }
    }
    
    @Override
    public void onVoiceActivityStarted() {
        if (callback != null) {
            callback.onVoiceActivityStarted();
        }
    }
    
    @Override
    public void onVoiceActivityEnded() {
        if (callback != null) {
            callback.onVoiceActivityEnded();
        }
    }
    
    @Override
    public void onAudioCaptureCompleted(String wakeWord, short[] audioData, int sampleRate) {
        if (callback != null) {
            callback.onAudioCaptureCompleted(wakeWord, audioData, sampleRate);
        }
    }
    
    /**
     * Enable or disable VAD processing
     *
     * @param enabled True to enable VAD, false to disable
     * @return true if the operation was successful
     */
    public boolean enableVAD(boolean enabled) {
        return enableVAD(nativeDetectorPtr, enabled);
    }

    // Native methods - implemented in C++
    private native long createWakeupDetector();
    private native boolean initializeDetector(long detectorPtr, String melModelPath, 
                                            String embModelPath, String[] wwModelPaths);
    private native boolean initializeVAD(long detectorPtr, String vadModelPath);
    private native boolean startDetector(long detectorPtr);
    private native void stopDetector(long detectorPtr);
    private native void processAudio(long detectorPtr, short[] audioData, int numSamples);
    private native boolean enableVAD(long detectorPtr, boolean enabled);
    private native void destroyWakeupDetector(long detectorPtr);
}