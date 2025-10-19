// This file is correct from the previous step. No changes needed.
// It uses raw pointers and catch_unwind for stability.
use crate::ImeEngine;
use std::ffi::{CStr, CString};
use libc::c_char;
use std::path::PathBuf;
use std::ptr;
use std::panic::{catch_unwind, AssertUnwindSafe};

static mut IME_ENGINE: *mut ImeEngine = ptr::null_mut();

fn get_dictionary_path() -> PathBuf {
    let mut path = dirs::data_local_dir()
        .or_else(dirs::home_dir)
        .expect("Could not find a valid home/data directory");
    path.push("nepali-smart-ime");
    path.push("user_dictionary.bin");
    path
}

#[no_mangle]
pub extern "C" fn nepali_ime_engine_init() {
    let result = catch_unwind(|| {
        unsafe {
            if !IME_ENGINE.is_null() { return; }
            let dict_path = get_dictionary_path();
            if let Some(parent) = dict_path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let engine = ImeEngine::from_file_or_new(dict_path.to_str().unwrap_or(""));
            IME_ENGINE = Box::into_raw(Box::new(engine));
            eprintln!("[Rust] Nepali IME Engine Initialized successfully.");
        }
    });
    if result.is_err() {
        eprintln!("[Rust FATAL] A panic occurred during IME engine initialization.");
        unsafe { IME_ENGINE = ptr::null_mut(); }
    }
}

#[no_mangle]
pub extern "C" fn nepali_ime_engine_destroy() {
    unsafe {
        if IME_ENGINE.is_null() { return; }
        let engine = Box::from_raw(IME_ENGINE);
        if let Err(e) = engine.save_dictionary() {
            eprintln!("[Rust ERR] Failed to save dictionary: {}", e);
        } else {
            eprintln!("[Rust] Dictionary saved successfully.");
        }
        IME_ENGINE = ptr::null_mut();
    }
}

unsafe fn get_engine_mut<'a>() -> Option<&'a mut ImeEngine> { IME_ENGINE.as_mut() }
unsafe fn get_engine<'a>() -> Option<&'a ImeEngine> { IME_ENGINE.as_ref() }

#[no_mangle]
pub extern "C" fn nepali_ime_get_suggestions(prefix: *const c_char) -> *mut c_char {
    let c_str = unsafe { CStr::from_ptr(prefix) };
    let roman_prefix = c_str.to_str().unwrap_or("");
    let result = catch_unwind(AssertUnwindSafe(|| {
        unsafe {
            if let Some(engine) = get_engine() {
                let suggestions = engine.get_suggestions(roman_prefix, 8);
                let json_suggestions: Vec<String> = suggestions.into_iter().map(|(s, _)| s).collect();
                return serde_json::to_string(&json_suggestions).unwrap_or_else(|_| "[]".to_string());
            }
        }
        "[]".to_string()
    }));
    let json_string = result.unwrap_or_else(|_| {
        eprintln!("[Rust FATAL] Panic in get_suggestions.");
        "[]".to_string()
    });
    CString::new(json_string).unwrap().into_raw()
}

#[no_mangle]
pub extern "C" fn nepali_ime_confirm_word(roman: *const c_char, nepali: *const c_char) {
    let roman_str = unsafe { CStr::from_ptr(roman) }.to_str().unwrap_or("");
    let nepali_str = unsafe { CStr::from_ptr(nepali) }.to_str().unwrap_or("");
    if !roman_str.is_empty() && !nepali_str.is_empty() {
        let _ = catch_unwind(AssertUnwindSafe(|| {
            unsafe { if let Some(engine) = get_engine_mut() { engine.user_confirms(roman_str, nepali_str); } }
        }));
    }
}

#[no_mangle]
pub extern "C" fn nepali_ime_free_string(s: *mut c_char) {
    if !s.is_null() { unsafe { let _ = CString::from_raw(s); } }
}