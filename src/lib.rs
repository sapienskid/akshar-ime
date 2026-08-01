// File: src/lib.rs

pub mod c_api;
pub mod core;
pub mod fuzzy;
pub mod learning;
pub mod persistence;

pub use crate::core::engine::ImeEngine;
