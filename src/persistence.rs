use crate::core::engine::ImeEngine;
use crate::core::trie::TrieBuilder;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Error, Write};
use std::path::Path;
use tempfile::NamedTempFile;

/// The serializable state of the application. We don't save the full engine,
/// just the parts that constitute learned user data.
#[derive(serde::Serialize, serde::Deserialize)]
struct SerializableState {
    trie_builder: TrieBuilder,
    context_model: crate::core::context::ContextModel,
}

pub fn save_to_disk(engine: &ImeEngine, path: &Path) -> Result<(), Error> {
    let parent_dir = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent_dir)?;

    let state = SerializableState {
        trie_builder: serde_json::from_str(&serde_json::to_string(&engine.trie_builder).unwrap()).unwrap(),
        context_model: serde_json::from_str(&serde_json::to_string(&engine.context_model).unwrap()).unwrap(),
    };

    let temp_file = NamedTempFile::new_in(parent_dir)?;
    let writer = BufWriter::new(&temp_file);

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