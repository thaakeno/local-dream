package io.github.xororz.localdream.service

import android.content.Context
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.IOException
import java.io.RandomAccessFile
import java.nio.ByteOrder
import java.util.UUID
import kotlin.math.roundToInt

/**
 * Offline audio importer for Breeze reference cloning.
 * Converts PCM WAV, MP3, M4A, OGG etc supported by Android codecs to
 * 24 kHz mono PCM16 WAV. Uses the FIRST 30 seconds, never truncates encoded
 * byte streams blindly. Caller owns the temporary output file.
 */
object BreezeAudioImporter {
    const val MAX_SECONDS = 30
    const val RATE = 24_000
    data class Prepared(val wav: File, val seconds: Float, val trimmed: Boolean)

    fun prepare(context: Context, source: Uri): Prepared {
        val out = File(context.cacheDir, "breeze_import_${UUID.randomUUID()}.wav")
        try {
            val mono: ShortArray
            var rate: Int
            var hadMore = false
            val wavPcm = tryPcmWav(context, source)
            if (wavPcm != null) {
                mono = wavPcm.first
                rate = wavPcm.second
                hadMore = wavPcm.third
            } else {
                val decoded = decodeCompressed(context, source)
                mono = decoded.first
                rate = decoded.second
                hadMore = decoded.third
            }
            require(rate > 0) { "Audio has no valid sample rate" }
            val required = (BreezeReferenceRecorder.MIN_SECONDS * rate)
            require(mono.size >= required) {
                "Choose at least 3 seconds of speech for a reliable voice reference"
            }
            val maxSamples = MAX_SECONDS * RATE
            val outSamples = minOf(maxSamples,
                (mono.size.toDouble() * RATE / rate).toInt())
            // Linear interpolation of decoded samples, with a cross-buffer
            // continuous phase; never changes pitch when input is 44.1/48 kHz.
            val wavSamples = ShortArray(outSamples)
            for (i in wavSamples.indices) {
                val index = i.toDouble() * rate / RATE
                val left = index.toInt().coerceIn(0, mono.lastIndex)
                val right = (left + 1).coerceAtMost(mono.lastIndex)
                val fraction = index - left
                val value = mono[left] * (1.0 - fraction) + mono[right] * fraction
                wavSamples[i] = value.roundToInt().coerceIn(-32768, 32767).toShort()
            }
            RandomAccessFile(out, "rw").use { f ->
                val bytes = wavSamples.size * 2
                f.writeBytes("RIFF")
                f.writeLE32(bytes + 36)
                f.writeBytes("WAVEfmt ")
                f.writeLE32(16); f.writeLE16(1); f.writeLE16(1)
                f.writeLE32(RATE); f.writeLE32(RATE * 2)
                f.writeLE16(2); f.writeLE16(16)
                f.writeBytes("data"); f.writeLE32(bytes)
                for (sample in wavSamples) f.writeLE16(sample.toInt())
            }
            return Prepared(out, outSamples / RATE.toFloat(),
                hadMore || (mono.size.toDouble() / rate > MAX_SECONDS))
        } catch (e: Exception) {
            out.delete()
            throw e
        }
    }

    private fun tryPcmWav(context: Context, uri: Uri): Triple<ShortArray, Int, Boolean>? {
        val input = context.contentResolver.openInputStream(uri) ?: throw IOException("Cannot open audio")
        input.buffered().use { stream ->
            val bytes = ByteArray(12)
            if (stream.read(bytes) != 12 ||
                String(bytes, 0, 4) != "RIFF" || String(bytes, 8, 4) != "WAVE") return null
            var format = 0
            var channels = 0
            var rate = 0
            var depth = 0
            while (true) {
                val header = ByteArray(8)
                if (stream.read(header) != 8) break
                val length = readLE32(header, 4)
                if (length < 0) throw IOException("Invalid WAV chunk")
                when (String(header, 0, 4)) {
                    "fmt " -> {
                        if (length < 16 || length > 4096) throw IOException("Invalid WAV format")
                        val data = ByteArray(length)
                        stream.readFully(data)
                        format = readLE16(data, 0)
                        channels = readLE16(data, 2)
                        rate = readLE32(data, 4)
                        depth = readLE16(data, 14)
                    }
                    "data" -> {
                        if (format != 1 || depth != 16 || channels !in 1..2 ||
                            rate !in 8000..192000) {
                            // WAV format unsupported; Android extractor may decode it.
                            return null
                        }
                        val frameBytes = channels * 2
                        val requested = MAX_SECONDS * rate
                        val framesAvailable = length / frameBytes
                        val take = minOf(requested, framesAvailable)
                        val mono = ShortArray(take)
                        val frame = ByteArray(frameBytes)
                        for (i in 0 until take) {
                            stream.readFully(frame)
                            var v = readLE16(frame, 0).toShort().toInt()
                            if (channels == 2) {
                                v = (v + readLE16(frame, 2).toShort().toInt()) / 2
                            }
                            mono[i] = v.toShort()
                        }
                        return Triple(mono, rate, framesAvailable > requested)
                    }
                    else -> stream.skipFully(length.toLong())
                }
                if (length % 2 == 1) stream.skipFully(1)
            }
        }
        return null
    }

    private fun decodeCompressed(context: Context, uri: Uri): Triple<ShortArray, Int, Boolean> {
        val extractor = MediaExtractor()
        var codec: MediaCodec? = null
        try {
            extractor.setDataSource(context, uri, null)
            val index = (0 until extractor.trackCount).firstOrNull {
                extractor.getTrackFormat(it).getString(MediaFormat.KEY_MIME)
                    ?.startsWith("audio/") == true
            } ?: throw IOException("Selected file has no audio track")
            extractor.selectTrack(index)
            val format = extractor.getTrackFormat(index)
            val mime = format.getString(MediaFormat.KEY_MIME)
                ?: throw IOException("Audio codec is missing")
            val decoder = MediaCodec.createDecoderByType(mime)
            codec = decoder
            // Request standard PCM16. Some codecs report floating-point output
            // despite the request; we handle that explicitly below.
            format.setInteger(MediaFormat.KEY_PCM_ENCODING, AudioFormat.ENCODING_PCM_16BIT)
            decoder.configure(format, null, null, 0)
            decoder.start()
            val output = ByteArrayOutputStream()
            val info = MediaCodec.BufferInfo()
            var reachedInputEnd = false
            var reachedOutputEnd = false
            var currentFormat = format
            var rate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            var channels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
            var truncated = false
            var stalls = 0
            while (!reachedOutputEnd && stalls < 150) {
                if (!reachedInputEnd) {
                    val inputIndex = decoder.dequeueInputBuffer(10_000)
                    if (inputIndex >= 0) {
                        val input = decoder.getInputBuffer(inputIndex)
                            ?: throw IOException("Decoder input unavailable")
                        val size = extractor.readSampleData(input, 0)
                        if (size < 0) {
                            decoder.queueInputBuffer(inputIndex, 0, 0, 0,
                                MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            reachedInputEnd = true
                        } else {
                            decoder.queueInputBuffer(inputIndex, 0, size,
                                extractor.sampleTime.coerceAtLeast(0), 0)
                            extractor.advance()
                        }
                    }
                }
                val outIndex = decoder.dequeueOutputBuffer(info, 10_000)
                when (outIndex) {
                    MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        currentFormat = decoder.outputFormat
                        rate = currentFormat.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                        channels = currentFormat.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                    }
                    MediaCodec.INFO_TRY_AGAIN_LATER -> stalls++
                    else -> if (outIndex >= 0) {
                        stalls = 0
                        val data = decoder.getOutputBuffer(outIndex)
                            ?: throw IOException("Decoder output unavailable")
                        data.order(ByteOrder.LITTLE_ENDIAN)
                        data.position(info.offset)
                        data.limit(info.offset + info.size)
                        val encoding = if (currentFormat.containsKey(MediaFormat.KEY_PCM_ENCODING))
                            currentFormat.getInteger(MediaFormat.KEY_PCM_ENCODING)
                        else AudioFormat.ENCODING_PCM_16BIT
                        require(channels in 1..8) { "Unsupported channel count" }
                        val sampleSize = if (encoding == AudioFormat.ENCODING_PCM_FLOAT) 4 else 2
                        val frames = data.remaining() / (sampleSize * channels)
                        val maxFrames = (MAX_SECONDS * rate).coerceAtMost(30 * 192000)
                        val already = output.size() / 2
                        val take = minOf(frames, maxFrames - already)
                        if (take <= 0) {
                            truncated = true
                            reachedOutputEnd = true
                        } else {
                            repeat(take) {
                                var value = 0.0
                                repeat(channels) {
                                    val one = if (encoding == AudioFormat.ENCODING_PCM_FLOAT)
                                        data.float.coerceIn(-1f, 1f).toDouble() * 32767.0
                                    else data.short.toDouble()
                                    value += one
                                }
                                val n = (value / channels).roundToInt().coerceIn(-32768, 32767)
                                output.write(n and 255)
                                output.write((n ushr 8) and 255)
                            }
                            if (take < frames) {
                                truncated = true
                                reachedOutputEnd = true
                            }
                        }
                        if ((info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0)
                            reachedOutputEnd = true
                        decoder.releaseOutputBuffer(outIndex, false)
                    }
                }
            }
            if (stalls >= 150) throw IOException("Audio decoder stalled")
            val bytes = output.toByteArray()
            val shorts = ShortArray(bytes.size / 2) { i ->
                readLE16(bytes, i * 2).toShort()
            }
            val durationUs = if (format.containsKey(MediaFormat.KEY_DURATION))
                format.getLong(MediaFormat.KEY_DURATION) else -1L
            return Triple(shorts, rate,
                truncated || (durationUs > MAX_SECONDS * 1_000_000L))
        } finally {
            runCatching { codec?.stop() }
            runCatching { codec?.release() }
            extractor.release()
        }
    }

    private fun readLE16(b: ByteArray, offset: Int): Int =
        (b[offset].toInt() and 255) or ((b[offset + 1].toInt() and 255) shl 8)
    private fun readLE32(b: ByteArray, offset: Int): Int =
        readLE16(b, offset) or (readLE16(b, offset + 2) shl 16)
    private fun java.io.InputStream.readFully(b: ByteArray) {
        var filled = 0
        while (filled < b.size) {
            val n = read(b, filled, b.size - filled)
            if (n < 0) throw IOException("Truncated WAV")
            filled += n
        }
    }
    private fun java.io.InputStream.skipFully(n: Long) {
        var remaining = n
        while (remaining > 0) {
            val skipped = skip(remaining)
            if (skipped <= 0) {
                if (read() == -1) throw IOException("Truncated WAV")
                remaining--
            } else remaining -= skipped
        }
    }
    private fun RandomAccessFile.writeLE16(n: Int) {
        write(n and 255); write((n ushr 8) and 255)
    }
    private fun RandomAccessFile.writeLE32(n: Int) {
        writeLE16(n); writeLE16(n ushr 16)
    }
}
