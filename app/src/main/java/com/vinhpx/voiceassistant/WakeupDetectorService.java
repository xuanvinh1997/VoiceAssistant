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
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;
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
    private static final int MAX_SPEECH_DURATION_SAMPLES = SAMPLE_RATE * 30; // 30 seconds max
    
    private final List<WakeupDetectorCallback> callbacks = new ArrayList<>();
    private final List<SpeechRecognitionListener> speechListeners = new ArrayList<>();
    private final Context context;
    private final Handler mainHandler;
    private final WakeupDetectorJNI detector;
    private final AtomicBoolean isRecording = new AtomicBoolean(false);
    
    private AudioRecord audioRecord;
    private short[] audioBuffer;
    private Thread recordingThread;
    
    // Audio capture for speech recognition
    private List<short[]> voiceActivityBuffer;
    private boolean isCapturingVoiceActivity = false;
    private boolean wakeWordDetectedDuringVoiceActivity = false;
    private String detectedWakeWord = null;
    private int capturedSamplesCount = 0;
    
    // Azure Speech SDK recognizer
    private AzureSpeechRecognizer speechRecognizer;
    private String azureSubscriptionKey;
    private String azureRegion;
    private String speechLanguage = "en-US"; // Default language
    
    // Add this field to store the last recognized speech text
    private String lastRecognizedText = null;
    
    // Add field for wake word detection timer
//    private Handler wakeWordDetectionHandler = new Handler(Looper.getMainLooper());
//    private Runnable wakeWordTimeoutRunnable;
    private static final long WAKE_WORD_HOLD_TIME_MS = 1000; // 1 second
    
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
        
        // Initialize buffer for voice activity
        voiceActivityBuffer = new ArrayList<>();
        
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
            String melModelPath = copyAssetToInternalStorage(melModelName);
            String embModelPath = copyAssetToInternalStorage(embModelName);
            
            String[] wakeWordModelPaths = new String[wakeWordModelNames.length];
            for (int i = 0; i < wakeWordModelNames.length; i++) {
                wakeWordModelPaths[i] = copyAssetToInternalStorage(wakeWordModelNames[i]);
            }
            
            return detector.initialize(melModelPath, embModelPath, wakeWordModelPaths);
        } catch (IOException e) {
            Log.e(TAG, "Error initializing models", e);
            return false;
        }
    }
    
    /**
     * Initialize the VAD (Voice Activity Detection) model
     * 
     * @param vadModelName Name of the VAD model file in assets
     * @return true if initialization succeeded
     */
    public boolean initializeVAD(String vadModelName) {
        try {
            String vadModelPath = copyAssetToInternalStorage(vadModelName);
            
            Log.i(TAG, "Initializing VAD with model: " + vadModelName);
            boolean result = detector.initializeVAD(vadModelPath);
            if (result) {
                Log.i(TAG, "VAD model initialized successfully");
            } else {
                Log.e(TAG, "Failed to initialize VAD model");
            }
            return result;
        } catch (IOException e) {
            Log.e(TAG, "Error initializing VAD model", e);
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
                // Process audio for wake word detection
                detector.processAudio(audioBuffer, readSize);
                
                // If we're in a voice activity session, store the audio for later processing
                if (isCapturingVoiceActivity && capturedSamplesCount < MAX_SPEECH_DURATION_SAMPLES) {
                    // Make a copy of the audio buffer to store
                    short[] bufferCopy = new short[readSize];
                    System.arraycopy(audioBuffer, 0, bufferCopy, 0, readSize);
                    
                    // Add to our voice activity buffer
                    voiceActivityBuffer.add(bufferCopy);
                    capturedSamplesCount += readSize;
                }
            }
        }
    }
    
    /**
     * Get the last recognized text from speech recognition
     * 
     * @return The last recognized text, or null if none
     */
    public String getLastRecognizedText() {
        return lastRecognizedText;
    }
    
    /**
     * Log text output using Android's Log system
     * 
     * @param text The text to log
     * @param priority Log priority level (use constants from android.util.Log)
     * @return true if logging succeeded
     */
    public boolean logTextOutput(String text, int priority) {
        if (text == null || text.isEmpty()) {
            return false;
        }
        
        try {
            // Format the log message with timestamp for better readability
            String timestamp = new SimpleDateFormat("HH:mm:ss", Locale.US).format(new Date());
            String logMessage = String.format("[%s] %s", timestamp, text);
            
            // Log with the specified priority
            switch (priority) {
                case Log.VERBOSE:
                    Log.v(TAG, logMessage);
                    break;
                case Log.DEBUG:
                    Log.d(TAG, logMessage);
                    break;
                case Log.INFO:
                    Log.i(TAG, logMessage);
                    break;
                case Log.WARN:
                    Log.w(TAG, logMessage);
                    break;
                case Log.ERROR:
                    Log.e(TAG, logMessage);
                    break;
                default:
                    Log.i(TAG, logMessage); // Default to INFO level
            }
            return true;
        } catch (Exception e) {
            Log.e(TAG, "Error logging text output", e);
            return false;
        }
    }
    
    /**
     * Log text output using Android's Log system with INFO priority
     * 
     * @param text The text to log
     * @return true if logging succeeded
     */
    public boolean logTextOutput(String text) {
        return logTextOutput(text, Log.INFO);
    }
    
    /**
     * Log the last recognized text from speech recognition with INFO priority
     * 
     * @return true if logging succeeded, false if there was no text to log
     */
    public boolean logLastRecognizedText() {
        if (lastRecognizedText != null && !lastRecognizedText.isEmpty()) {
            return logTextOutput(lastRecognizedText);
        }
        return false;
    }
    
    /**
     * Log the last recognized text with specified priority
     * 
     * @param priority Log priority level (use constants from android.util.Log)
     * @return true if logging succeeded, false if there was no text to log
     */
    public boolean logLastRecognizedText(int priority) {
        if (lastRecognizedText != null && !lastRecognizedText.isEmpty()) {
            return logTextOutput(lastRecognizedText, priority);
        }
        return false;
    }
    
    // WakeupDetectorCallback implementation

    @Override
    public void onWakeWordDetected(String wakeWord) {
        // Dispatch to all callbacks on main thread
        mainHandler.post(() -> {
            Log.i(TAG, "Wake word detected: " + wakeWord);
            
            // Always mark wake word as detected, regardless of current voice activity state
            wakeWordDetectedDuringVoiceActivity = true;
            detectedWakeWord = wakeWord;
            
            // Start voice activity capture immediately when wake word is detected
            isCapturingVoiceActivity = true;
            voiceActivityBuffer.clear();
            capturedSamplesCount = 0;
            
            // Enable VAD in the native detector to start listening for voice activity
            detector.enableVAD(true);
            
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
        
        // We should already have capture started from onWakeWordDetected
        // but this is a safety check in case VAD starts before wake word detection
        if (!isCapturingVoiceActivity) {
            isCapturingVoiceActivity = true;
            // We don't clear the buffer here as it might contain audio from the wake word
        }
        
        mainHandler.post(() -> {
            for (WakeupDetectorCallback callback : new ArrayList<>(callbacks)) {
                callback.onVoiceActivityStarted();
            }
        });
    }
    
    @Override
    public void onVoiceActivityEnded() {
        Log.d("VADDebug", "Voice activity ENDED detected");
        
        // Process captured audio if a wake word was detected during this voice activity
        if (wakeWordDetectedDuringVoiceActivity && !voiceActivityBuffer.isEmpty()) {
            Log.d(TAG, "Processing voice activity audio - captured " + capturedSamplesCount + " samples with wake word: " + detectedWakeWord);
            // Stop capturing before processing to prevent additional audio being added during processing
            isCapturingVoiceActivity = false;
            processVoiceActivityAudio();
            // Reset wake word flag after processing
            wakeWordDetectedDuringVoiceActivity = false;
        } else {
            // Clear the buffer if no wake word was detected
            Log.d(TAG, "No wake word was detected during voice activity or buffer is empty (wakeWordDetected=" + 
                   wakeWordDetectedDuringVoiceActivity + ", bufferEmpty=" + voiceActivityBuffer.isEmpty() + 
                   ", capturedSamples=" + capturedSamplesCount + ")");
            voiceActivityBuffer.clear();
            capturedSamplesCount = 0;
            isCapturingVoiceActivity = false;
            wakeWordDetectedDuringVoiceActivity = false;
        }
        
        mainHandler.post(() -> {
            for (WakeupDetectorCallback callback : new ArrayList<>(callbacks)) {
                callback.onVoiceActivityEnded();
            }
        });
    }
    
    /**
     * Process the audio captured during voice activity and send to Azure Speech
     */
    private void processVoiceActivityAudio() {
        // Calculate total audio size
        int totalSize = capturedSamplesCount;
        if (totalSize <= 0) {
            Log.w(TAG, "No voice activity audio captured");
            return;
        }
        
        Log.d(TAG, "Processing voice activity audio: combining " + voiceActivityBuffer.size() + 
              " chunks into " + totalSize + " samples");
        
        // Create a single audio buffer from all captured chunks
        short[] capturedAudio = new short[totalSize];
        int position = 0;
        
        // Copy all buffer chunks into the single array
        for (short[] chunk : voiceActivityBuffer) {
            int chunkSize = Math.min(chunk.length, totalSize - position);
            if (chunkSize <= 0) break;
            
            System.arraycopy(chunk, 0, capturedAudio, position, chunkSize);
            position += chunkSize;
        }
        
        Log.d(TAG, "Audio processing complete: combined " + position + " of " + totalSize + 
              " samples, sending to onAudioCaptureCompleted");
              
        final short[] finalAudio = capturedAudio;
        // Ensure wakeWord is never null - use a placeholder if needed
        final String finalWakeWord = detectedWakeWord != null ? detectedWakeWord : "unknown";
        
        // Clear buffers
        voiceActivityBuffer.clear();
        capturedSamplesCount = 0;
        
        // Send the captured audio for processing
        mainHandler.post(() -> {
            onAudioCaptureCompleted(finalWakeWord, finalAudio, SAMPLE_RATE);
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
                    
                    // Add extra debugging to track if this is actually being called
                    if (audioData != null && audioData.length > 0) {
                        Log.d(TAG, "Audio data is valid: " + audioData.length + " samples");
                        
                        // Check if we have any listeners registered
                        Log.d(TAG, "Number of speech recognition listeners: " + speechListeners.size());
                        
                        // Start the recognition process
                        speechRecognizer.recognizeSpeech(audioData, sampleRate);
                    } else {
                        Log.e(TAG, "Audio data is invalid or empty");
                    }
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