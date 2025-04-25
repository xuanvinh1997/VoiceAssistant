package com.vinhpx.voiceassistant;

/**
 * Callback interface for wake word detection events.
 */
public interface WakeupDetectorCallback {
    /**
     * Called when a wake word is detected.
     * 
     * @param wakeWord The name of the detected wake word
     */
    void onWakeWordDetected(String wakeWord);
    
    /**
     * Called when a detection score is available.
     * 
     * @param wakeWord The name of the wake word model
     * @param score The detection score (0.0-1.0)
     * @param threshold The current threshold
     * @param activation The current activation level
     * @param triggerLevel The trigger level required for detection
     */
    default void onDetectionScoreUpdate(String wakeWord, float score, float threshold, int activation, int triggerLevel) {
        // Default empty implementation - optional to implement
    }
    
    /**
     * Called when voice activity is detected.
     */
    default void onVoiceActivityStarted() {
        // Default empty implementation - optional to implement
    }
    
    /**
     * Called when voice activity has ended.
     */
    default void onVoiceActivityEnded() {
        // Default empty implementation - optional to implement
    }
    
    /**
     * Called when collected audio after wake word detection is ready.
     * 
     * @param wakeWord The name of the wake word that triggered the audio capture
     * @param audioData The collected audio data as PCM 16-bit samples
     * @param sampleRate The sample rate of the audio (typically 16000 Hz)
     */
    default void onAudioCaptureCompleted(String wakeWord, short[] audioData, int sampleRate) {
        // Default empty implementation - optional to implement
    }
}