package com.vinhpx.voiceassistant.speech;

import android.content.Context;
import android.util.Log;

import com.microsoft.cognitiveservices.speech.SpeechConfig;
import com.microsoft.cognitiveservices.speech.SpeechRecognizer;
import com.microsoft.cognitiveservices.speech.audio.AudioConfig;
import com.microsoft.cognitiveservices.speech.audio.PullAudioInputStream;
import com.microsoft.cognitiveservices.speech.audio.AudioInputStream;
import com.microsoft.cognitiveservices.speech.audio.PullAudioInputStreamCallback;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Handles speech recognition using Azure Speech SDK
 */
public class AzureSpeechRecognizer {
    private static final String TAG = "AzureSpeechRecognizer";
    
    private final ExecutorService executor = Executors.newSingleThreadExecutor();
    private final List<SpeechRecognitionListener> listeners = new ArrayList<>();
    private SpeechConfig speechConfig;
    
    /**
     * Initialize the Azure Speech recognizer
     *
     * @param subscriptionKey Azure Speech Service subscription key
     * @param region Azure Speech Service region (e.g., "westus")
     * @param language Language to recognize (e.g., "en-US")
     */
    public AzureSpeechRecognizer(String subscriptionKey, String region, String language) {
        try {
            speechConfig = SpeechConfig.fromSubscription(subscriptionKey, region);
            speechConfig.setSpeechRecognitionLanguage(language);
        } catch (Exception e) {
            Log.e(TAG, "Error initializing Azure Speech SDK: " + e.getMessage(), e);
        }
    }
    
    /**
     * Add a listener for speech recognition events
     *
     * @param listener The listener to add
     */
    public void addListener(SpeechRecognitionListener listener) {
        if (listener != null && !listeners.contains(listener)) {
            listeners.add(listener);
        }
    }
    
    /**
     * Remove a listener
     *
     * @param listener The listener to remove
     */
    public void removeListener(SpeechRecognitionListener listener) {
        listeners.remove(listener);
    }
    
    /**
     * Recognize speech from PCM audio data
     *
     * @param audioData PCM audio data as 16-bit samples
     * @param sampleRate Sample rate of the audio (typically 16000 Hz)
     */
    public Future<Void> recognizeSpeech(short[] audioData, int sampleRate) {
        if (speechConfig == null) {
            notifyError("Speech recognizer not initialized properly");
            return null;
        }
        
        return executor.submit(() -> {
            try {
                notifyRecognizing();
                
                // Create a pull audio input stream from the audio data
                PullAudioInputStream inputStream = AudioInputStream.createPullStream(
                        new AudioDataReader(audioData), 
                        AudioStreamFormat.create16kHz16BitMonoPcm());
                
                // Create audio config from the input stream
                AudioConfig audioConfig = AudioConfig.fromStreamInput(inputStream);
                
                // Create speech recognizer
                try (SpeechRecognizer recognizer = new SpeechRecognizer(speechConfig, audioConfig)) {
                    // Add recognition event handlers
                    recognizer.recognizing.addEventListener((o, e) -> {
                        notifyPartialResult(e.getResult().getText());
                    });
                    
                    recognizer.recognized.addEventListener((o, e) -> {
                        notifyResult(e.getResult().getText());
                    });
                    
                    recognizer.canceled.addEventListener((o, e) -> {
                        notifyError("Recognition canceled: " + e.getErrorDetails());
                    });
                    
                    // Start recognition
                    var result = recognizer.recognizeOnceAsync().get();
                    Log.i(TAG, "Recognition result: " + result.getText());
                }
                
                notifyCompleted();
            } catch (Exception e) {
                Log.e(TAG, "Error during speech recognition: " + e.getMessage(), e);
                notifyError("Recognition error: " + e.getMessage());
            }
            
            return null;
        });
    }
    
    /**
     * Release resources used by the recognizer
     */
    public void release() {
        executor.shutdown();
        if (speechConfig != null) {
            speechConfig.close();
            speechConfig = null;
        }
        listeners.clear();
    }
    
    /**
     * Notify listeners that recognition is starting
     */
    private void notifyRecognizing() {
        for (SpeechRecognitionListener listener : new ArrayList<>(listeners)) {
            listener.onRecognizing();
        }
    }
    
    /**
     * Notify listeners of a partial recognition result
     *
     * @param text The partial result text
     */
    private void notifyPartialResult(String text) {
        for (SpeechRecognitionListener listener : new ArrayList<>(listeners)) {
            listener.onPartialResult(text);
        }
    }
    
    /**
     * Notify listeners of a final recognition result
     *
     * @param text The final result text
     */
    private void notifyResult(String text) {
        for (SpeechRecognitionListener listener : new ArrayList<>(listeners)) {
            listener.onResult(text);
        }
    }
    
    /**
     * Notify listeners of a recognition error
     *
     * @param error The error message
     */
    private void notifyError(String error) {
        for (SpeechRecognitionListener listener : new ArrayList<>(listeners)) {
            listener.onError(error);
        }
    }
    
    /**
     * Notify listeners that recognition is completed
     */
    private void notifyCompleted() {
        for (SpeechRecognitionListener listener : new ArrayList<>(listeners)) {
            listener.onCompleted();
        }
    }
    
    /**
     * Helper class for audio format specification
     */
    private static class AudioStreamFormat {
        static com.microsoft.cognitiveservices.speech.audio.AudioStreamFormat create16kHz16BitMonoPcm() {
            return com.microsoft.cognitiveservices.speech.audio.AudioStreamFormat.getWaveFormatPCM(16000, (short)16, (short)1);
        }
    }
    
    /**
     * Callback to read audio data for the Speech SDK
     */
    private static class AudioDataReader extends PullAudioInputStreamCallback {
        private final short[] audioData;
        private int position = 0;
        
        AudioDataReader(short[] audioData) {
            this.audioData = audioData;
        }
        
        @Override
        public int read(byte[] buffer) {
            if (position >= audioData.length) {
                return 0; // End of stream
            }
            
            // Calculate how many samples we can copy
            int samplesToRead = Math.min(buffer.length / 2, audioData.length - position);
            
            // Convert short samples to bytes (little-endian)
            for (int i = 0; i < samplesToRead; i++) {
                short sample = audioData[position + i];
                buffer[i * 2] = (byte) (sample & 0xFF);
                buffer[i * 2 + 1] = (byte) ((sample >> 8) & 0xFF);
            }
            
            position += samplesToRead;
            return samplesToRead * 2; // Return number of bytes read
        }
        
        @Override
        public void close() {
            // Nothing to close
        }
    }
}