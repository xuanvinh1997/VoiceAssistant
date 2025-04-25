package com.vinhpx.voiceassistant.speech;

/**
 * Listener interface for speech recognition events
 */
public interface SpeechRecognitionListener {
    /**
     * Called when recognition is starting
     */
    default void onRecognizing() {}
    
    /**
     * Called when a partial result is available
     * 
     * @param text The partial result text
     */
    default void onPartialResult(String text) {}
    
    /**
     * Called when a final result is available
     * 
     * @param text The final result text
     */
    void onResult(String text);
    
    /**
     * Called when an error occurs during recognition
     * 
     * @param error The error message
     */
    default void onError(String error) {}
    
    /**
     * Called when recognition is completed
     */
    default void onCompleted() {}
}