use std::fs::{create_dir_all, OpenOptions};
use std::path::PathBuf;
use std::io::{Seek, SeekFrom, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use http::HeaderMap;
use jni::objects::{JClass, JString};
use jni::sys::{jint, jlong, jstring};
use jni::JNIEnv;
use xet::xet_session::{XetFileInfo, XetSession, XetSessionBuilder};

static CANCEL_REQUESTED: AtomicBool = AtomicBool::new(false);
static ACTIVE_SESSION: Mutex<Option<XetSession>> = Mutex::new(None);
static ACTIVE_PROGRESS_BYTES: AtomicU64 = AtomicU64::new(0);
static ACTIVE_PROGRESS_TOTAL: AtomicU64 = AtomicU64::new(0);
static ACTIVE_PROGRESS_SPEED_BPS: AtomicU64 = AtomicU64::new(0);
static ACTIVE_TRANSFER_BYTES: AtomicU64 = AtomicU64::new(0);
static ACTIVE_TRANSFER_TOTAL: AtomicU64 = AtomicU64::new(0);
static ACTIVE_TRANSFER_SPEED_BPS: AtomicU64 = AtomicU64::new(0);
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

fn configure_writable_runtime(cache_dir: &str) -> Result<(), String> {
    let root = PathBuf::from(cache_dir);
    let xet_cache = root.join("xet");
    let hf_home = root.join("home");
    let xdg_cache = root.join("xdg");
    let tmp_dir = root.join("tmp");

    for dir in [&root, &xet_cache, &hf_home, &xdg_cache, &tmp_dir] {
        create_dir_all(dir)
            .map_err(|e| format!("Cannot create Xet runtime directory {}: {e}", dir.display()))?;
    }

    // Android app processes do not have a normal writable Unix home directory.
    // Point every Xet/runtime scratch location at app-private storage before
    // constructing the Xet session.
    std::env::set_var("HF_XET_CACHE", &xet_cache);
    std::env::set_var("HF_HOME", &hf_home);
    std::env::set_var("XDG_CACHE_HOME", &xdg_cache);
    std::env::set_var("HOME", &hf_home);
    std::env::set_var("TMPDIR", &tmp_dir);

    // Keep mobile disk usage tiny: downloads do not need the optional chunk
    // cache, and console logging avoids Xet creating its default logs directory.
    std::env::set_var("HF_XET_LOG_DEST", "");

    Ok(())
}

fn bytes_as_mib_env(bytes: u64) -> String {
    const MIB: u64 = 1024 * 1024;
    let mib = bytes.saturating_add(MIB - 1) / MIB;
    format!("{}mb", mib.max(1))
}

fn configure_mobile_runtime(
    memory_budget_bytes: u64,
    min_concurrency: i32,
    initial_concurrency: i32,
    max_concurrency: i32,
) {
    #[cfg(target_os = "android")]
    std::env::set_var("SSL_CERT_DIR", "/system/etc/security/cacerts");

    std::env::remove_var("HF_XET_HIGH_PERFORMANCE");
    std::env::remove_var("HF_XET_HP");

    let budget = memory_budget_bytes.max(1);

    let min_c = min_concurrency.max(1);
    let max_c = max_concurrency.max(min_c);
    let initial_c = initial_concurrency.clamp(min_c, max_c);

    // Keep all sizes derived from the live Android memory/concurrency budget. A single
    // reconstruction lane should not need to fill hundreds of MB before progress becomes
    // observable, especially on normal phone Wi-Fi.
    let lanes = (max_c as u64).max(1);
    let per_lane = (budget / lanes).max(1);
    let base_buffer = (budget / 2).max(per_lane);
    let per_file_buffer = (budget / 4).max(per_lane);
    let prefetch_buffer = per_lane;
    let min_fetch = (per_lane / 8).max(1);
    let max_fetch = per_lane.saturating_mul(2).max(min_fetch);

    std::env::set_var("HF_XET_CLIENT_ENABLE_ADAPTIVE_CONCURRENCY", "1");
    std::env::set_var("HF_XET_CLIENT_AC_MIN_DOWNLOAD_CONCURRENCY", min_c.to_string());
    std::env::set_var("HF_XET_CLIENT_AC_INITIAL_DOWNLOAD_CONCURRENCY", initial_c.to_string());
    std::env::set_var("HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY", max_c.to_string());
    std::env::set_var("HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS", "1");

    std::env::set_var("HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE", bytes_as_mib_env(base_buffer));
    std::env::set_var("HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_PERFILE_SIZE", bytes_as_mib_env(per_file_buffer));
    std::env::set_var("HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT", bytes_as_mib_env(budget));
    std::env::set_var("HF_XET_RECONSTRUCTION_MIN_PREFETCH_BUFFER", bytes_as_mib_env(prefetch_buffer));
    std::env::set_var("HF_XET_RECONSTRUCTION_MIN_RECONSTRUCTION_FETCH_SIZE", bytes_as_mib_env(min_fetch));
    std::env::set_var("HF_XET_RECONSTRUCTION_MAX_RECONSTRUCTION_FETCH_SIZE", bytes_as_mib_env(max_fetch));
    std::env::set_var("HF_XET_TELEMETRY_ENABLED", "0");
}
fn run_download(
    hash: String,
    size: u64,
    refresh_url: String,
    dest_path: String,
    cache_dir: String,
    offset: u64,
    memory_budget_bytes: u64,
    min_concurrency: i32,
    initial_concurrency: i32,
    max_concurrency: i32,
) -> Result<i32, String> {
    if offset > size {
        return Err(format!("resume offset {offset} is larger than file size {size}"));
    }
    if offset == size {
        return Ok(0);
    }

    configure_writable_runtime(&cache_dir)?;
    configure_mobile_runtime(
        memory_budget_bytes,
        min_concurrency,
        initial_concurrency,
        max_concurrency,
    );
    CANCEL_REQUESTED.store(false, Ordering::Release);
    ACTIVE_PROGRESS_BYTES.store(offset, Ordering::Release);
    ACTIVE_PROGRESS_TOTAL.store(size, Ordering::Release);
    ACTIVE_PROGRESS_SPEED_BPS.store(0, Ordering::Release);
    ACTIVE_TRANSFER_BYTES.store(0, Ordering::Release);
    ACTIVE_TRANSFER_TOTAL.store(0, Ordering::Release);
    ACTIVE_TRANSFER_SPEED_BPS.store(0, Ordering::Release);
    set_error("");

    let session = XetSessionBuilder::new()
        .build()
        .map_err(|e| format!("Xet session creation failed: {e}"))?;

    if let Ok(mut active) = ACTIVE_SESSION.lock() {
        *active = Some(session.clone());
    }

    let file_info = XetFileInfo::new(hash.clone(), size);

    let result = if offset == 0 {
        let group = session
            .new_file_download_group()
            .map_err(|e| format!("Xet download group creation failed: {e}"))?
            .with_token_refresh_url(refresh_url.clone(), HeaderMap::new())
            .build_blocking()
            .map_err(|e| format!("Xet authentication failed: {e}"))?;

        let observer = group.clone();
        let monitor_stop = Arc::new(AtomicBool::new(false));
        let monitor_stop_worker = monitor_stop.clone();
        let monitor = thread::spawn(move || {
            while !monitor_stop_worker.load(Ordering::Acquire) {
                let progress = observer.progress();
                ACTIVE_PROGRESS_BYTES.store(progress.total_bytes_completed, Ordering::Release);
                ACTIVE_PROGRESS_TOTAL.store(progress.total_bytes, Ordering::Release);
                ACTIVE_PROGRESS_SPEED_BPS.store(
                    progress.total_bytes_completion_rate.unwrap_or(0.0).max(0.0) as u64,
                    Ordering::Release,
                );
                ACTIVE_TRANSFER_BYTES.store(
                    progress.total_transfer_bytes_completed,
                    Ordering::Release,
                );
                ACTIVE_TRANSFER_TOTAL.store(progress.total_transfer_bytes, Ordering::Release);
                ACTIVE_TRANSFER_SPEED_BPS.store(
                    progress
                        .total_transfer_bytes_completion_rate
                        .unwrap_or(0.0)
                        .max(0.0) as u64,
                    Ordering::Release,
                );
                thread::sleep(Duration::from_millis(200));
            }
        });

        let download_result = (|| -> Result<i32, String> {
            group
                .download_file_to_path_blocking(file_info.clone(), PathBuf::from(&dest_path))
                .map_err(|e| format!("Xet file download start failed: {e}"))?;
            group
                .finish_blocking()
                .map_err(|e| format!("Xet download failed: {e}"))?;

            if CANCEL_REQUESTED.load(Ordering::Acquire) {
                return Ok(1);
            }

            ACTIVE_PROGRESS_BYTES.store(size, Ordering::Release);
            ACTIVE_PROGRESS_TOTAL.store(size, Ordering::Release);
            Ok(0)
        })();

        monitor_stop.store(true, Ordering::Release);
        let _ = monitor.join();
        download_result
    } else {
        let group = session
            .new_download_stream_group()
            .map_err(|e| format!("Xet download group creation failed: {e}"))?
            .with_token_refresh_url(refresh_url, HeaderMap::new())
            .build_blocking()
            .map_err(|e| format!("Xet authentication failed: {e}"))?;

        let mut stream = group
            .download_stream_blocking(file_info, Some(offset..size))
            .map_err(|e| format!("Xet stream creation failed: {e}"))?;

        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .open(&dest_path)
            .map_err(|e| format!("Cannot open partial file: {e}"))?;

        file.set_len(offset)
            .map_err(|e| format!("Cannot truncate partial file: {e}"))?;
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| format!("Cannot seek partial file: {e}"))?;

        let mut written = offset;
        (|| -> Result<i32, String> {
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
                        written = written.saturating_add(bytes.len() as u64).min(size);
                        ACTIVE_PROGRESS_BYTES.store(written, Ordering::Release);
                        ACTIVE_PROGRESS_TOTAL.store(size, Ordering::Release);
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
            ACTIVE_PROGRESS_BYTES.store(size, Ordering::Release);
            Ok(0)
        })()
    };

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
    cache_dir: JString,
    offset: jlong,
    memory_budget_bytes: jlong,
    min_concurrency: jint,
    initial_concurrency: jint,
    max_concurrency: jint,
) -> jint {
    let result = (|| {
        let hash = from_jstring(&mut env, hash)?;
        let refresh_url = from_jstring(&mut env, refresh_url)?;
        let dest_path = from_jstring(&mut env, dest_path)?;
        let cache_dir = from_jstring(&mut env, cache_dir)?;
        if size < 0 || offset < 0 || memory_budget_bytes <= 0 {
            return Err("invalid size/offset/runtime memory budget".to_string());
        }
        run_download(
            hash,
            size as u64,
            refresh_url,
            dest_path,
            cache_dir,
            offset as u64,
            memory_budget_bytes as u64,
            min_concurrency,
            initial_concurrency,
            max_concurrency,
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
pub extern "system" fn Java_io_github_xororz_localdream_service_XetNative_nativeProgressBytes(
    _env: JNIEnv,
    _class: JClass,
) -> jlong {
    ACTIVE_PROGRESS_BYTES.load(Ordering::Acquire).min(i64::MAX as u64) as jlong
}

#[no_mangle]
pub extern "system" fn Java_io_github_xororz_localdream_service_XetNative_nativeProgressTotalBytes(
    _env: JNIEnv,
    _class: JClass,
) -> jlong {
    ACTIVE_PROGRESS_TOTAL.load(Ordering::Acquire).min(i64::MAX as u64) as jlong
}

#[no_mangle]
pub extern "system" fn Java_io_github_xororz_localdream_service_XetNative_nativeProgressBytesPerSecond(
    _env: JNIEnv,
    _class: JClass,
) -> jlong {
    ACTIVE_PROGRESS_SPEED_BPS.load(Ordering::Acquire).min(i64::MAX as u64) as jlong
}

#[no_mangle]
pub extern "system" fn Java_io_github_xororz_localdream_service_XetNative_nativeTransferBytes(
    _env: JNIEnv,
    _class: JClass,
) -> jlong {
    ACTIVE_TRANSFER_BYTES.load(Ordering::Acquire).min(i64::MAX as u64) as jlong
}

#[no_mangle]
pub extern "system" fn Java_io_github_xororz_localdream_service_XetNative_nativeTransferTotalBytes(
    _env: JNIEnv,
    _class: JClass,
) -> jlong {
    ACTIVE_TRANSFER_TOTAL.load(Ordering::Acquire).min(i64::MAX as u64) as jlong
}

#[no_mangle]
pub extern "system" fn Java_io_github_xororz_localdream_service_XetNative_nativeTransferBytesPerSecond(
    _env: JNIEnv,
    _class: JClass,
) -> jlong {
    ACTIVE_TRANSFER_SPEED_BPS.load(Ordering::Acquire).min(i64::MAX as u64) as jlong
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
