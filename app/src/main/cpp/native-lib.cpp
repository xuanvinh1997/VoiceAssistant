#include <jni.h>
#include <string>
#include <android/log.h>
#include "wakeup_detector.h"

#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, "NativeLib", __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "NativeLib", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "NativeLib", __VA_ARGS__)

extern "C" JNIEXPORT jstring JNICALL
Java_com_vinhpx_voiceassistant_MainActivity_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Wake-up Word Service Initialized";
    return env->NewStringUTF(hello.c_str());
}

// The WakeupDetectorJNI functions are implemented in wakeup_detector.cpp