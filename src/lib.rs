// File: src/lib.rs

#[cfg(not(target_arch = "wasm32"))]
pub mod c_api;
#[cfg(target_arch = "wasm32")]
pub mod wasm;

pub mod core;
pub mod fuzzy;
pub mod learning;
pub mod persistence;

pub use crate::core::engine::ImeEngine;
