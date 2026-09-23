use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use http::HeaderMap;
use jni::objects::{JClass, JString};
use jni::sys::{jint, jlong, jstring};
use jni::JNIEnv;
use xet::xet_session::{XetFileInfo, XetSession, XetSessionBuilder};

static CANCEL_REQUESTED: AtomicBool = AtomicBool::new(false);
static ACTIVE_SESSION: Mutex<Option<XetSession>> = Mutex::new(None);
static LAST_ERROR: Mutex<String> = Mutex::new(String::new());

fn set_error(message: impl Into<String>) {
    if let Ok(mut error) = LAST_ERROR.lock() {
        *error = message.into();
    }
}

fn from_jstring(env: &mut JNIEnv<'_>, value: JString<'_>) -> Result<String, String> {
    env.get_string(&value)
        .map(|s| s.into())
        .map_err(|e| format!("JNI string conversion failed: {e}"))
}

fn configure_mobile_profile(profile: i32) {
    // Never enable the desktop-oriented HP preset on Android.
    std::env::remove_var("HF_XET_HIGH_PERFORMANCE");
    std::env::remove_var("HF_XET_HP");

    let (initial, max, buffer, per_file, limit, prefetch) = match profile {
        // Hot / constrained.
        1 => ("1", "2", "64mb", "24mb", "96mb", "48mb"),
        // Warm.
        2 => ("2", "4", "96mb", "32mb", "160mb", "64mb"),
        // Cool/default. Still intentionally far below desktop Xet defaults.
        _ => ("3", "6", "128mb", "48mb", "224mb", "96mb"),
    };

    std::env::set_var("HF_XET_CLIENT_ENABLE_ADAPTIVE_CONCURRENCY", "1");
    std::env::set_var("HF_XET_CLIENT_AC_MIN_DOWNLOAD_CONCURRENCY", "1");
    std::env::set_var("HF_XET_CLIENT_AC_INITIAL_DOWNLOAD_CONCURRENCY", initial);
    std::env::set_var("HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY", max);
    std::env::set_var("HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS", "1");

    // Explicitly bound reconstruction memory for a phone. The upstream defaults
    // can scale into multi-GB buffers on large-memory machines.
    std::env::set_var("HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE", buffer);
    std::env::set_var("HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_PERFILE_SIZE", per_file);
    std::env::set_var("HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT", limit);
    std::env::set_var("HF_XET_RECONSTRUCTION_MIN_PREFETCH_BUFFER", prefetch);
    std::env::set_var("HF_XET_RECONSTRUCTION_MIN_RECONSTRUCTION_FETCH_SIZE", "32mb");
    std::env::set_var("HF_XET_RECONSTRUCTION_MAX_RECONSTRUCTION_FETCH_SIZE", "512mb");
    std::env::set_var("HF_XET_TELEMETRY_ENABLED", "0");
}

fn run_download(
    hash: String,
    size: u64,
    refresh_url: String,
    dest_path: String,
    offset: u64,
    profile: i32,
) -> Result<i32, String> {
    if offset > size {
        return Err(format!("resume offset {offset} is larger than file size {size}"));
    }
    if offset == size {
        return Ok(0);
    }

    configure_mobile_profile(profile);
    CANCEL_REQUESTED.store(false, Ordering::Release);

    let session = XetSessionBuilder::new()
        .build()
        .map_err(|e| format!("Xet session creation failed: {e}"))?;

    let group = session
        .new_download_stream_group()
        .map_err(|e| format!("Xet download group creation failed: {e}"))?
        .with_token_refresh_url(refresh_url, HeaderMap::new())
        .build_blocking()
        .map_err(|e| format!("Xet authentication failed: {e}"))?;

    if let Ok(mut active) = ACTIVE_SESSION.lock() {
        *active = Some(session.clone());
    }

    let result = (|| -> Result<i32, String> {
        let file_info = XetFileInfo::new(hash, size);
        let mut stream = group
            .download_stream_blocking(file_info, Some(offset..size))
            .map_err(|e| format!("Xet stream creation failed: {e}"))?;

        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .open(&dest_path)
            .map_err(|e| format!("Cannot open partial file: {e}"))?;

        // The Kotlin side passes the last known contiguous byte offset. Trim any
        // stale tail before appending the resumed range.
        file.set_len(offset)
            .map_err(|e| format!("Cannot truncate partial file: {e}"))?;
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| format!("Cannot seek partial file: {e}"))?;

        loop {
            if CANCEL_REQUESTED.load(Ordering::Acquire) {
                stream.cancel();
                let _ = session.abort();
                return Ok(1);
            }

            match stream.blocking_next() {
                Ok(Some(bytes)) => {
                    file.write_all(&bytes)
                        .map_err(|e| format!("Writing Xet data failed: {e}"))?;
                }
                Ok(None) => break,
                Err(e) => {
                    if CANCEL_REQUESTED.load(Ordering::Acquire) {
                        return Ok(1);
                    }
                    return Err(format!("Xet download failed: {e}"));
                }
            }
        }

        file.flush()
            .map_err(|e| format!("Flushing Xet data failed: {e}"))?;
        file.set_len(size)
            .map_err(|e| format!("Finalizing Xet file failed: {e}"))?;

        // Streaming groups intentionally have no finish() API. Reaching
        // blocking_next() == None means the requested range is complete.
        Ok(0)
    })();

    if let Ok(mut active) = ACTIVE_SESSION.lock() {
        *active = None;
    }
    result
}

#[no_mangle]
pub extern "system" fn Java_io_github_xororz_localdream_service_XetNative_nativeDownload(
    mut env: JNIEnv,
    _class: JClass,
    hash: JString,
    size: jlong,
    refresh_url: JString,
    dest_path: JString,
    offset: jlong,
    profile: jint,
) -> jint {
    let result = (|| {
        let hash = from_jstring(&mut env, hash)?;
        let refresh_url = from_jstring(&mut env, refresh_url)?;
        let dest_path = from_jstring(&mut env, dest_path)?;
        if size < 0 || offset < 0 {
            return Err("negative size/offset".to_string());
        }
        run_download(
            hash,
            size as u64,
            refresh_url,
            dest_path,
            offset as u64,
            profile,
        )
    })();

    match result {
        Ok(code) => code as jint,
        Err(error) => {
            set_error(error);
            -1
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_io_github_xororz_localdream_service_XetNative_nativeCancel(
    _env: JNIEnv,
    _class: JClass,
) {
    CANCEL_REQUESTED.store(true, Ordering::Release);
    if let Ok(active) = ACTIVE_SESSION.lock() {
        if let Some(session) = active.as_ref() {
            let _ = session.abort();
        }
    }
}

#[no_mangle]
pub extern "system" fn Java_io_github_xororz_localdream_service_XetNative_nativeLastError(
    env: JNIEnv,
    _class: JClass,
) -> jstring {
    let message = LAST_ERROR
        .lock()
        .map(|s| s.clone())
        .unwrap_or_else(|_| "unknown Xet error".to_string());
    match env.new_string(message) {
        Ok(value) => value.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}
