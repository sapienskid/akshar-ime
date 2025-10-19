use nepali_smart_ime::ImeEngine;
use std::io::{self, BufRead, Write};
use std::fs::File;
use std::path::{ PathBuf};

// This function now reliably gets the correct path for any user.
fn get_dictionary_path() -> PathBuf {
    let mut path = dirs::config_dir().expect("Could not find config directory");
    path.push("nepali-smart-ime");
    path.push("user_dictionary.bin");
    path
}

fn get_log_path() -> PathBuf {
    let mut path = PathBuf::from("target");
    path.push("nepali_ime_rust.log");
    path
}

fn log(message: &str) {
    if let Ok(mut file) = File::options().create(true).append(true).open(get_log_path()) {
        let _ = writeln!(file, "{}", message);
    }
}

fn main() -> io::Result<()> {
    // Clear old log file
    let _ = std::fs::remove_file(get_log_path());
    log("--- Nepali IME Rust Engine Starting ---");

    let dict_path = get_dictionary_path();
    if let Some(parent) = dict_path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            log(&format!("Error creating config dir: {}", e));
        }
    }
    
    let mut engine = ImeEngine::from_file_or_new(dict_path.to_str().unwrap());
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut preedit = String::new();

    for line in stdin.lock().lines() {
        let input = line?;
        log(&format!("Rust <- '{:?}'", input));
        let parts: Vec<&str> = input.split_whitespace().collect();
        let command = parts.get(0).cloned().unwrap_or("");

        match command {
            "PROCESS_KEY_EVENT" => {
                let key_val: u32 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
                let key_code: u32 = parts.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);

                if handle_key_event(key_val, key_code, &mut preedit, &mut engine, &mut stdout)? {
                    update_ibus(&preedit, &engine.get_suggestions(&preedit, 5), &mut stdout)?;
                }
            }
            "EXIT" => {
                log("Rust: Received EXIT, saving dictionary.");
                if let Err(e) = engine.save_dictionary() { log(&format!("Error saving dict: {}", e)); }
                break;
            }
            _ => { log("Rust: Received unknown command."); }
        }
    }
    log("Rust: Shutting down.");
    Ok(())
}

fn handle_key_event(key_val: u32, _key_code: u32, preedit: &mut String, engine: &mut ImeEngine, stdout: &mut io::Stdout) -> io::Result<bool> {
    let mut needs_update = true;
    match key_val {
        32 | 65293 | 65289 => { // Space, Enter, Tab
            if !preedit.is_empty() {
                let suggestions = engine.get_suggestions(preedit, 1);
                let to_commit = if !suggestions.is_empty() {
                    suggestions[0].0.clone()
                } else {
                    engine.romanizer.generate_candidates(preedit).get(0).cloned().unwrap_or_default()
                };

                if !to_commit.is_empty() {
                    let cmd = format!("COMMIT_TEXT {}", to_commit);
                    log(&format!("Rust -> '{:?}'", cmd));
                    writeln!(stdout, "{}", cmd)?;
                    engine.user_confirms(preedit, &to_commit);
                }
                preedit.clear();
            } else {
                let cmd = "COMMIT_TEXT  "; // Commit a single space
                log(&format!("Rust -> '{:?}'", cmd));
                writeln!(stdout, "{}", cmd)?;
            }
        }
        65288 => { preedit.pop(); } // Backspace
        65307 => { preedit.clear(); } // Escape
        _ => {
            if let Some(c) = std::char::from_u32(key_val) {
                if c.is_ascii_alphabetic() {
                    preedit.push(c.to_ascii_lowercase());
                } else if c.is_ascii_digit() || c.is_ascii_punctuation() {
                    preedit.push(c);
                } else {
                    needs_update = false;
                }
            } else { needs_update = false; }
        }
    }
    Ok(needs_update)
}

fn update_ibus(preedit: &str, suggestions: &[(String, u64)],  stdout: &mut io::Stdout) -> io::Result<()> {
    let cmd = format!("UPDATE_PREEDIT_TEXT {} 0 true", preedit);
    log(&format!("Rust -> '{:?}'", cmd));
    writeln!(stdout, "{}", cmd)?;
    
    log("Rust -> 'UPDATE_LOOKUP_TABLE'");
    writeln!(stdout, "UPDATE_LOOKUP_TABLE")?;

    if suggestions.is_empty() {
        log("Rust -> 'HIDE_LOOKUP_TABLE'");
        writeln!(stdout, "HIDE_LOOKUP_TABLE")?;
    } else {
        for (i, (word, _)) in suggestions.iter().enumerate() {
            let cmd = format!("ADD_CANDIDATE {} '{}' {}", i, word, i);
            log(&format!("Rust -> '{:?}'", cmd));
            writeln!(stdout, "{}", cmd)?;
        }
        log("Rust -> 'SHOW_LOOKUP_TABLE'");
        writeln!(stdout, "SHOW_LOOKUP_TABLE")?;
    }
    stdout.flush()
}