package io.github.xororz.localdream.service

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Process
import android.os.SystemClock
import androidx.core.content.ContextCompat
import java.io.File
import java.io.RandomAccessFile
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.abs

/**
 * On-device, permission-gated, single-speaker voice recording.
 * PCM 24 kHz, mono, signed 16-bit -> standard RIFF WAV accepted directly by
 * Breeze's existing local /v1/voices endpoint. No MediaRecorder AAC conversion
 * and no third-party storage or network access.
 */
class BreezeReferenceRecorder(private val context: Context) {
    companion object {
        const val MIN_SECONDS = 3
        const val RECOMMENDED_SECONDS = 12
        const val MAX_SECONDS = 30
        private const val RATE = 24_000
    }
    private val capturing = AtomicBoolean(false)
    val peak = AtomicInteger(0)
    private var recorder: AudioRecord? = null
    private var worker: Thread? = null
    private var rawFile: File? = null
    private var startMs: Long = 0

    val recording: Boolean get() = capturing.get()
    val elapsedMs: Long get() = if (recording) SystemClock.elapsedRealtime() - startMs else 0L

    @SuppressLint("MissingPermission")
    fun start() {
        check(!recording) { "Already recording" }
        check(ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
              PackageManager.PERMISSION_GRANTED) { "Microphone permission required" }
        val minimum = AudioRecord.getMinBufferSize(RATE,
            AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        require(minimum > 0) { "Device does not support 24 kHz mono PCM recording" }
        val source = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT,
            maxOf(minimum * 2, 8192))
        if (source.state != AudioRecord.STATE_INITIALIZED) {
            source.release()
            error("Microphone could not be initialized")
        }
        val raw = File.createTempFile("breeze_reference_", ".pcm", context.cacheDir)
        try {
            source.startRecording()
            check(source.recordingState == AudioRecord.RECORDSTATE_RECORDING) {
                "Microphone is not recording"
            }
        } catch (e: Exception) {
            source.release()
            raw.delete()
            throw e
        }
        recorder = source
        rawFile = raw
        peak.set(0)
        startMs = SystemClock.elapsedRealtime()
        capturing.set(true)
        worker = Thread({
            Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO)
            val buffer = ByteArray(4096)
            try {
                raw.outputStream().buffered().use { output ->
                    while (capturing.get()) {
                        val count = source.read(buffer, 0, buffer.size, AudioRecord.READ_BLOCKING)
                        if (count <= 0) {
                            if (count != 0 && capturing.get()) break
                            continue
                        }
                        output.write(buffer, 0, count)
                        var max = 0
                        var i = 0
                        while (i + 1 < count) {
                            val pcm = ((buffer[i+1].toInt() shl 8) or
                                (buffer[i].toInt() and 0xff)).toShort().toInt()
                            max = maxOf(max, abs(pcm))
                            i += 2
                        }
                        peak.set(max)
                    }
                }
            } finally {
                capturing.set(false)
            }
        }, "breeze-mic-pcm").apply { isDaemon = true; start() }
    }

    /** Run on Dispatchers.IO. Failure or an undersized clip deletes the draft. */
    fun stop(): File {
        val raw = rawFile ?: error("No voice recording to finish")
        val old = recorder
        capturing.set(false)
        runCatching { old?.stop() }
        worker?.join(2500)
        if (worker?.isAlive == true) {
            runCatching { old?.release() }
            worker?.join(1000)
        } else runCatching { old?.release() }
        recorder = null
        worker = null
        rawFile = null
        check(!capturing.get()) { "Recording thread did not stop" }
        try {
            val audioBytes = raw.length()
            val seconds = audioBytes.toDouble() / (RATE * 2.0)
            require(seconds >= MIN_SECONDS) {
                "Record at least $MIN_SECONDS seconds of clear speech"
            }
            require(seconds <= MAX_SECONDS + 0.5) {
                "Reference recording exceeds $MAX_SECONDS seconds"
            }
            val result = File(context.cacheDir, raw.nameWithoutExtension + ".wav")
            RandomAccessFile(result, "rw").use { wav ->
                val payload = audioBytes.toInt()
                wav.writeBytes("RIFF")
                wav.writeLEInt(payload + 36)
                wav.writeBytes("WAVEfmt ")
                wav.writeLEInt(16)
                wav.writeLEShort(1) // PCM
                wav.writeLEShort(1) // mono
                wav.writeLEInt(RATE)
                wav.writeLEInt(RATE * 2)
                wav.writeLEShort(2)
                wav.writeLEShort(16)
                wav.writeBytes("data")
                wav.writeLEInt(payload)
                raw.inputStream().use { input ->
                    val buffer = ByteArray(64 * 1024)
                    while (true) {
                        val n = input.read(buffer)
                        if (n < 0) break
                        wav.write(buffer, 0, n)
                    }
                }
            }
            return result
        } finally {
            raw.delete()
        }
    }

    fun discard() {
        capturing.set(false)
        runCatching { recorder?.stop() }
        runCatching { recorder?.release() }
        recorder = null
        runCatching { worker?.join(1000) }
        worker = null
        rawFile?.delete()
        rawFile = null
    }
}

private fun RandomAccessFile.writeLEInt(value: Int) {
    write(value and 0xff)
    write((value ushr 8) and 0xff)
    write((value ushr 16) and 0xff)
    write((value ushr 24) and 0xff)
}
private fun RandomAccessFile.writeLEShort(value: Int) {
    write(value and 0xff)
    write((value ushr 8) and 0xff)
}
