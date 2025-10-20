// File: src/persistence.rs
use crate::core::engine::ImeEngine;
use crate::core::trie::TrieBuilder;
use crate::fuzzy::symspell::SymSpell;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Error, ErrorKind}; // <-- ADDED ErrorKind for clarity
use std::path::Path;
use tempfile::NamedTempFile;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct SerializableState {
    trie_builder: TrieBuilder,
    context_model: crate::core::context::ContextModel,
    symspell: SymSpell,
}

pub fn save_to_disk(engine: &ImeEngine, path: &Path) -> Result<(), Error> {
    let parent_dir = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent_dir)?;

    let state = SerializableState {
        trie_builder: engine.trie_builder.clone(),
        context_model: engine.context_model.clone(),
        symspell: engine.symspell.clone(),
    };

    let temp_file = NamedTempFile::new_in(parent_dir)?;
    let writer = BufWriter::new(&temp_file);

    // --- THE FIX IS HERE ---
    // Changed `std.io` to `std::io` and simplified the call using the `use` statement.
    // This now correctly maps the bincode error into a standard std::io::Error.
    bincode::serialize_into(writer, &state)
        .map_err(|e| Error::new(ErrorKind::Other, e))?;

    temp_file.persist(path)?;
    Ok(())
}

pub fn load_from_disk(path: &Path) -> Result<ImeEngine, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let state: SerializableState = bincode::deserialize_from(reader)?;
    
    // Reconstruct the engine from the stateful components.
    // The stateless engines (romanizer, phonetic, learning) are created by ImeEngine::new().
    let mut engine = ImeEngine::new();
    engine.trie_builder = state.trie_builder;
    engine.context_model = state.context_model;
    engine.symspell = state.symspell;
    
    Ok(engine)
}