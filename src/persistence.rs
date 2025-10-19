// File: src/persistence.rs
use crate::core::engine::ImeEngine;
use crate::core::trie::TrieBuilder;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Error};
use std::path::Path;
use tempfile::NamedTempFile;

/// The serializable state of the application.
/// It now derives Clone to make saving trivial.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct SerializableState {
    trie_builder: TrieBuilder,
    context_model: crate::core::context::ContextModel, // <-- CORRECT
}

pub fn save_to_disk(engine: &ImeEngine, path: &Path) -> Result<(), Error> {
    let parent_dir = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent_dir)?;

    // THE FIX: No more serde_json. Just clone the data directly.
    // This is safe, fast, and works with our data structures.
    let state = SerializableState {
        trie_builder: engine.trie_builder.clone(),
        context_model: engine.context_model.clone(),
    };

    let temp_file = NamedTempFile::new_in(parent_dir)?;
    let writer = BufWriter::new(&temp_file);

    // Bincode serializes the clean, cloned state without any issues.
    bincode::serialize_into(writer, &state)
        .map_err(|e| Error::new(std::io::ErrorKind::Other, e))?;

    temp_file.persist(path)?;
    Ok(())
}

pub fn load_from_disk(path: &Path) -> Result<ImeEngine, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let state: SerializableState = bincode::deserialize_from(reader)?;
    
    let mut engine = ImeEngine::new();
    engine.trie_builder = state.trie_builder;
    engine.context_model = state.context_model;
    
    Ok(engine)
}