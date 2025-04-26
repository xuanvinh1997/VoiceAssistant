package com.vinhpx.voiceassistant

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.isVisible
import androidx.core.view.updateLayoutParams
import com.vinhpx.voiceassistant.databinding.ActivityMainBinding
import com.vinhpx.voiceassistant.speech.SpeechRecognitionListener

class MainActivity : AppCompatActivity(), WakeupDetectorCallback {

    private lateinit var binding: ActivityMainBinding
    private lateinit var wakeupDetectorService: WakeupDetectorService
    
    private var isDetectionRunning = false
    private val RECORD_AUDIO_PERMISSION_CODE = 101
    
    // Track detection metric colors
    private val normalColor = 0xFF4CAF50.toInt() // Green
    private val warningColor = 0xFFFF9800.toInt() // Orange
    private val activatedColor = 0xFFFF5252.toInt() // Red

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        // Set up UI
        binding.sampleText.text = stringFromJNI()
        
        // Initialize wake-up detector service
        wakeupDetectorService = WakeupDetectorService(applicationContext)
        wakeupDetectorService.addCallback(this)
        wakeupDetectorService.configureAzureSpeech(
            "88ae64471dbf4933923b7ecf27151576",
            "southeastasia",
            "en-US"  // or your preferred language
        )
        
        // Add speech recognition listener to receive speech-to-text results
        wakeupDetectorService.addSpeechRecognitionListener(SpeechListener())
        
        // Set up button listeners
        binding.startButton.setOnClickListener {
            if (!isDetectionRunning) {
                startDetection()
            }
        }

        binding.stopButton.setOnClickListener {
            if (isDetectionRunning) {
                stopDetection()
            }
        }
        
        // Request audio permission
        requestAudioPermission()
        
        // Initialize detection score display
        binding.thresholdMarker.isVisible = false
    }
    
    override fun onDestroy() {
        super.onDestroy()
        wakeupDetectorService.release()
    }
    
    private fun requestAudioPermission() {
        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.RECORD_AUDIO),
                RECORD_AUDIO_PERMISSION_CODE
            )
        }
    }
    
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        
        if (requestCode == RECORD_AUDIO_PERMISSION_CODE) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                Toast.makeText(this, "Audio permission granted", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "Audio permission denied - cannot detect wake words", Toast.LENGTH_LONG).show()
                binding.startButton.isEnabled = false
            }
        }
    }
    
    private fun startDetection() {
        binding.statusText.text = "Initializing..."
        
        // Add logging to check VAD status
        Log.d("VADDebug", "Starting detection - checking VAD functionality")
        
        // Initialize with models from assets folder
        // Note: These files need to be placed in the assets folder
        val melModelName = "models/melspectrogram.onnx"
        val embModelName = "models/embedding_model.onnx"
        val wakeWordModels = arrayOf("models/alexa_v0.1.onnx")
        
        // Add VAD model initialization
        val vadModelName = "models/vad.onnx"
        Log.d("VADDebug", "VAD model path: $vadModelName")
        
        if (wakeupDetectorService.initialize(melModelName, embModelName, wakeWordModels)) {
            // Initialize VAD model
            if (wakeupDetectorService.initializeVAD(vadModelName)) {
                Log.d("VADDebug", "VAD model initialized successfully")
            } else {
                Log.e("VADDebug", "Failed to initialize VAD model")
                binding.statusText.text = "Note: VAD not available."
            }

            if (wakeupDetectorService.start()) {
                isDetectionRunning = true
                binding.statusText.text = "Listening for wake words..."
                binding.startButton.isEnabled = false
                binding.stopButton.isEnabled = true
                
                // Reset score display
                resetScoreDisplay()
            } else {
                binding.statusText.text = "Failed to start detection."
                Log.e("VADDebug", "Failed to start detector")
            }
        } else {
            binding.statusText.text = "Failed to initialize models."
            Log.e("VADDebug", "Failed to initialize models")
        }
    }
    
    private fun stopDetection() {
        wakeupDetectorService.stop()
        isDetectionRunning = false
        binding.statusText.text = "Detection stopped."
        binding.startButton.isEnabled = true
        binding.stopButton.isEnabled = false
        
        // Reset score display
        resetScoreDisplay()
    }
    
    private fun resetScoreDisplay() {
        binding.scoreValue.text = "0.000"
        binding.activationValue.text = "Activation: 0/0"
        binding.scoreMeter.progress = 0
        binding.scoreMeter.setProgressTintList(ContextCompat.getColorStateList(this, android.R.color.holo_green_light))
        binding.thresholdMarker.isVisible = false
    }
    
    // Callback when a wake word is detected
    override fun onWakeWordDetected(wakeWord: String) {
        val message = "Wake word detected: $wakeWord"
        binding.statusText.text = message
        binding.detectionHistory.append("$message\n")
        
        // Show that we're listening for user speech
        binding.statusText.text = "Listening... (after '$wakeWord')"
        
        // Audio will be automatically captured and sent to Azure Speech SDK
        // by the WakeupDetectorService's onAudioCaptureCompleted method
    }
    
    // Override audio capture completion to handle results from speech recognition
    override fun onAudioCaptureCompleted(wakeWord: String, audioData: ShortArray, sampleRate: Int) {
        binding.statusText.text = "Processing speech..."
        binding.detectionHistory.append("Audio captured: ${audioData.size} samples\n")
    }
    
    // Add speech recognition listener to handle speech-to-text results
    inner class SpeechListener : SpeechRecognitionListener {
        override fun onRecognizing() {
            binding.statusText.text = "Recognizing speech..."
        }
        
        override fun onPartialResult(text: String) {
            binding.statusText.text = "Heard: $text"
        }
        
        override fun onResult(text: String) {
            val resultMessage = "\"$text\""
            binding.statusText.text = resultMessage
            binding.detectionHistory.append("$resultMessage\n")
        }
        
        override fun onError(error: String) {
            binding.statusText.text = "Speech recognition error"
            binding.detectionHistory.append("Speech error: $error\n")
        }
        
        override fun onCompleted() {
            // Recognition completed
        }
    }

    /**
     * A native method that is implemented by the 'voiceassistant' native library,
     * which is packaged with this application.
     */
    external fun stringFromJNI(): String

    companion object {
        // Used to load the 'voiceassistant' library on application startup.
        init {
            System.loadLibrary("voiceassistant")
        }
    }

    // Override voice activity detection callbacks
    override fun onVoiceActivityStarted() {
        Log.d("VADDebug", "MainActivity: Voice activity started")
        runOnUiThread {
            binding.statusText.text = "Voice activity detected..."
            binding.vadStatus.visibility = View.VISIBLE
            binding.vadStatus.setBackgroundColor(activatedColor)
            binding.detectionHistory.append("Voice activity started\n")
        }
    }
    
    override fun onVoiceActivityEnded() {
        Log.d("VADDebug", "MainActivity: Voice activity ended")
        runOnUiThread {
            binding.statusText.text = "Listening for wake words..."
            binding.vadStatus.visibility = View.VISIBLE
            binding.vadStatus.setBackgroundColor(normalColor)
            binding.detectionHistory.append("Voice activity ended\n")
        }
    }
}