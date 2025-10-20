// File: src/lib.rs

pub mod core;
pub mod learning;
pub mod persistence;
pub mod c_api;
pub mod fuzzy;

pub use crate::core::engine::ImeEngine;