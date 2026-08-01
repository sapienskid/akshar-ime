// File: src/persistence.rs
use crate::core::engine::ImeEngine;
use crate::core::trie::Trie;
use crate::core::types::TransliterationModel;
use crate::fuzzy::symspell::SymSpell;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Error, ErrorKind};
use std::path::Path;
use tempfile::NamedTempFile;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct SerializableState {
    trie: Trie,
    context_model: crate::core::context::ContextModel,
    symspell: SymSpell,
    #[serde(default)]
    transliteration_model: TransliterationModel,
}

pub fn save_to_disk(engine: &ImeEngine, path: &Path) -> Result<(), Error> {
    let parent_dir = path.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(parent_dir)?;

    let state = SerializableState {
        trie: engine.trie.clone(),
        context_model: engine.context_model.clone(),
        symspell: engine.symspell.clone(),
        transliteration_model: engine.transliteration_model.clone(),
    };

    let temp_file = NamedTempFile::new_in(parent_dir)?;
    let writer = BufWriter::new(&temp_file);

    bincode::serialize_into(writer, &state).map_err(|e| Error::new(ErrorKind::Other, e))?;

    temp_file.persist(path)?;
    Ok(())
}

pub fn load_from_disk(path: &Path) -> Result<ImeEngine, Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let state: SerializableState = bincode::deserialize_from(reader)?;

    let mut engine = ImeEngine::new();
    engine.trie = state.trie;
    engine.context_model = state.context_model;
    engine.symspell = state.symspell;
    engine.transliteration_model = state.transliteration_model;

    Ok(engine)
}
