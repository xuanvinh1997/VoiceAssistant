package com.vinhpx.voiceassistant;

import android.content.Context;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

import com.vinhpx.voiceassistant.speech.AzureSpeechRecognizer;
import com.vinhpx.voiceassistant.speech.SpeechRecognitionListener;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Service to manage wake word detection using the native detector.
 */
public class WakeupDetectorService implements WakeupDetectorCallback {
    private static final String TAG = "WakeupDetectorService";
    
    // Audio configurations
    private static final int SAMPLE_RATE = 16000; // 16kHz
    private static final int CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO;
    private static final int AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT;
    private static final int BUFFER_SIZE_FACTOR = 2; // Multiplier for minimum buffer size
    
    private final List<WakeupDetectorCallback> callbacks = new ArrayList<>();
    private final List<SpeechRecognitionListener> speechListeners = new ArrayList<>();
    private final Context context;
    private final Handler mainHandler;
    private final WakeupDetectorJNI detector;
    private final AtomicBoolean isRecording = new AtomicBoolean(false);
    
    private AudioRecord audioRecord;
    private short[] audioBuffer;
    private Thread recordingThread;
    
    // Azure Speech SDK recognizer
    private AzureSpeechRecognizer speechRecognizer;
    private String azureSubscriptionKey;
    private String azureRegion;
    private String speechLanguage = "en-US"; // Default language
    
    // Add this field to store the last recognized speech text
    private String lastRecognizedText = null;
    
    /**
     * Create a new WakeupDetectorService
     * 
     * @param context The application context
     */
    public WakeupDetectorService(Context context) {
        this.context = context;
        this.mainHandler = new Handler(Looper.getMainLooper());
        this.detector = new WakeupDetectorJNI();
        this.detector.setCallback(this);
        
        int minBufferSize = AudioRecord.getMinBufferSize(
                SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT);
        audioBuffer = new short[minBufferSize];
        
        Log.i(TAG, "WakeupDetectorService created with buffer size: " + minBufferSize);
    }
    
    /**
     * Initialize the detector with the ONNX models
     * 
     * @param melModelName Name of the mel spectrogram model file in assets
     * @param embModelName Name of the embedding model file in assets
     * @param wakeWordModelNames Array of wake word model files in assets
     * @return true if initialization succeeded
     */
    public boolean initialize(String melModelName, String embModelName, String[] wakeWordModelNames) {
        try {
            // Copy model files to internal storage for native access
            String melModelPath = copyAssetToInternalStorage(melModelName);
            String embModelPath = copyAssetToInternalStorage(embModelName);
            
            String[] wakeWordModelPaths = new String[wakeWordModelNames.length];
            for (int i = 0; i < wakeWordModelNames.length; i++) {
                wakeWordModelPaths[i] = copyAssetToInternalStorage(wakeWordModelNames[i]);
            }
            
            return detector.initialize(melModelPath, embModelPath, wakeWordModelPaths);
        } catch (Exception e) {
            Log.e(TAG, "Error initializing detector", e);
            return false;
        }
    }

    
    /**
     * Configure Azure Speech Recognition
     * 
     * @param subscriptionKey Azure Speech Service subscription key
     * @param region Azure Speech Service region (e.g., "westus")
     * @param language Recognition language (e.g., "en-US")
     */
    public void configureAzureSpeech(String subscriptionKey, String region, String language) {
        this.azureSubscriptionKey = subscriptionKey;
        this.azureRegion = region;
        this.speechLanguage = language != null ? language : "en-US";
        
        // Initialize Azure Speech recognizer if keys are provided
        if (subscriptionKey != null && !subscriptionKey.isEmpty() && 
            region != null && !region.isEmpty()) {
            
            if (speechRecognizer != null) {
                speechRecognizer.release();
            }
            
            speechRecognizer = new AzureSpeechRecognizer(subscriptionKey, region, speechLanguage);
            
            // Add internal listener to track recognized text
            speechRecognizer.addListener(new SpeechRecognitionListener() {
                @Override
                public void onRecognizing() {
                    // Recognition is starting
                }
                
                @Override
                public void onPartialResult(String text) {
                    // Update with partial results
                    lastRecognizedText = text;
                }
                
                @Override
                public void onResult(String text) {
                    // Update with final results
                    lastRecognizedText = text;
                    Log.d(TAG, "Azure speech recognition result: \"" + text + "\"");
                }
                
                @Override
                public void onError(String error) {
                    Log.e(TAG, "Azure speech recognition error: " + error);
                }
                
                @Override
                public void onCompleted() {
                    // Recognition is completed
                }
            });
            
            Log.i(TAG, "Azure Speech SDK initialized for region: " + region);
        }
    }
    
    /**
     * Start wake word detection
     * 
     * @return true if starting succeeded
     */
    public boolean start() {
        if (isRecording.get()) {
            Log.w(TAG, "Already running");
            return true;
        }
        
        // Start the native detector
        if (!detector.start()) {
            Log.e(TAG, "Failed to start native detector");
            return false;
        }
        
        // Start audio recording
        try {
            audioRecord = new AudioRecord(MediaRecorder.AudioSource.MIC, 
                    SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT, audioBuffer.length * BUFFER_SIZE_FACTOR);
            
            if (audioRecord.getState() != AudioRecord.STATE_INITIALIZED) {
                Log.e(TAG, "AudioRecord initialization failed");
                return false;
            }
            
            audioRecord.startRecording();
            isRecording.set(true);
            
            // Start processing thread
            recordingThread = new Thread(this::processAudioRunnable);
            recordingThread.start();
            
            Log.i(TAG, "WakeupDetectorService started");
            return true;
        } catch (Exception e) {
            Log.e(TAG, "Error starting audio recording", e);
            stop();
            return false;
        }
    }
    
    /**
     * Stop wake word detection
     */
    public void stop() {
        isRecording.set(false);
        
        // Stop native detector
        detector.stop();
        
        // Stop and release AudioRecord
        if (audioRecord != null) {
            if (audioRecord.getState() == AudioRecord.STATE_INITIALIZED) {
                try {
                    audioRecord.stop();
                } catch (IllegalStateException e) {
                    Log.w(TAG, "AudioRecord was not recording", e);
                }
            }
            audioRecord.release();
            audioRecord = null;
        }
        
        // Wait for recording thread to finish
        if (recordingThread != null) {
            try {
                recordingThread.join(1000);
            } catch (InterruptedException e) {
                Log.w(TAG, "Interrupted while waiting for recording thread", e);
                Thread.currentThread().interrupt();
            }
            recordingThread = null;
        }
        
        Log.i(TAG, "WakeupDetectorService stopped");
    }
    
    /**
     * Release all resources
     */
    public void release() {
        stop();
        detector.release();
        callbacks.clear();
        
        // Release Azure Speech resources
        if (speechRecognizer != null) {
            speechRecognizer.release();
            speechRecognizer = null;
        }
        
        Log.i(TAG, "WakeupDetectorService released");
    }
    
    /**
     * Add a callback for wake word detection events
     * 
     * @param callback The callback to add
     */
    public void addCallback(WakeupDetectorCallback callback) {
        if (callback != null && !callbacks.contains(callback)) {
            callbacks.add(callback);
        }
    }
    
    /**
     * Remove a callback
     * 
     * @param callback The callback to remove
     */
    public void removeCallback(WakeupDetectorCallback callback) {
        callbacks.remove(callback);
    }
    
    /**
     * Add a speech recognition listener
     * 
     * @param listener The listener to add
     */
    public void addSpeechRecognitionListener(SpeechRecognitionListener listener) {
        if (listener != null && !speechListeners.contains(listener)) {
            speechListeners.add(listener);
            
            // Add to the Azure recognizer if it exists
            if (speechRecognizer != null) {
                speechRecognizer.addListener(listener);
            }
        }
    }
    
    /**
     * Remove a speech recognition listener
     * 
     * @param listener The listener to remove
     */
    public void removeSpeechRecognitionListener(SpeechRecognitionListener listener) {
        speechListeners.remove(listener);
        
        // Remove from the Azure recognizer if it exists
        if (speechRecognizer != null) {
            speechRecognizer.removeListener(listener);
        }
    }
    
    /**
     * Runnable for audio processing thread
     */
    private void processAudioRunnable() {
        while (isRecording.get()) {
            int readSize = audioRecord.read(audioBuffer, 0, audioBuffer.length);
            
            if (readSize > 0) {
                detector.processAudio(audioBuffer, readSize);
            }
        }
    }
    
    // WakeupDetectorCallback implementation

    @Override
    public void onWakeWordDetected(String wakeWord) {
        // Dispatch to all callbacks on main thread
        mainHandler.post(() -> {
            Log.i(TAG, "Wake word detected: " + wakeWord);
            for (WakeupDetectorCallback callback : new ArrayList<>(callbacks)) {
                callback.onWakeWordDetected(wakeWord);
            }
        });
    }
    
    @Override
    public void onDetectionScoreUpdate(String wakeWord, float score, float threshold, 
                                       int activation, int triggerLevel) {
        // Only dispatch significant updates (reduces UI load)
        if (score > threshold * 0.5 || activation > 0) {
            mainHandler.post(() -> {
                for (WakeupDetectorCallback callback : new ArrayList<>(callbacks)) {
                    callback.onDetectionScoreUpdate(wakeWord, score, threshold, activation, triggerLevel);
                }
            });
        }
    }
    
    @Override
    public void onVoiceActivityStarted() {
        Log.d("VADDebug", "Voice activity STARTED detected");
        mainHandler.post(() -> {
            for (WakeupDetectorCallback callback : callbacks) {
                callback.onVoiceActivityStarted();
            }
        });
    }
    
    @Override
    public void onVoiceActivityEnded() {
        Log.d("VADDebug", "Voice activity ENDED detected");
        mainHandler.post(() -> {
            for (WakeupDetectorCallback callback : callbacks) {
                callback.onVoiceActivityEnded();
            }
        });
    }
    
    @Override
    public void onAudioCaptureCompleted(String wakeWord, short[] audioData, int sampleRate) {
        mainHandler.post(() -> {
            Log.i(TAG, "Audio capture completed: " + audioData.length + " samples at " + sampleRate + " Hz");
            
            // Notify callbacks
            for (WakeupDetectorCallback callback : new ArrayList<>(callbacks)) {
                callback.onAudioCaptureCompleted(wakeWord, audioData, sampleRate);
            }
            
            // Send to Azure Speech SDK if configured
            if (speechRecognizer != null) {
                try {
                    Log.i(TAG, "Sending captured audio to Azure Speech SDK");
                    speechRecognizer.recognizeSpeech(audioData, sampleRate);
                } catch (Exception e) {
                    Log.e(TAG, "Error sending audio to Azure Speech SDK", e);
                }
            } else {
                Log.w(TAG, "Azure Speech SDK not configured, skipping speech recognition");
            }
        });
    }
    

    
    /**
     * Copy an asset file to internal storage
     * 
     * @param assetName The name of the asset file
     * @return The path to the copied file
     * @throws IOException If an I/O error occurs
     */
    private String copyAssetToInternalStorage(String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        
        // Create parent directories if they don't exist
        File parentDir = file.getParentFile();
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs();
        }
        
        // Copy asset file to internal storage
        try (FileOutputStream outputStream = new FileOutputStream(file)) {
            byte[] buffer = new byte[1024];
            int read;
            
            try (var inputStream = context.getAssets().open(assetName)) {
                while ((read = inputStream.read(buffer)) != -1) {
                    outputStream.write(buffer, 0, read);
                }
            }
        }
        
        return file.getAbsolutePath();
    }
}