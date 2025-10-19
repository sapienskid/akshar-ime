// src/lib.rs

pub mod core;
pub mod learning;
pub mod persistence;
pub mod c_api; // <-- ADD THIS LINE
pub use crate::core::engine::ImeEngine;