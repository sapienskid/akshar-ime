use ime_core::ImeEngine;
use std::io::{stdin, stdout, Write};

const DICTIONARY_PATH: &str = "user_dictionary.bin";

fn main() {
    let mut engine = ImeEngine::from_file_or_new(DICTIONARY_PATH);
    let mut preedit = String::new();

    println!("Nepali Smart IME (Full Version). Type 'exit' to save and quit.");
    println!("---------------------------------------------------------------");

    loop {
        let suggestions = engine.get_suggestions(&preedit, 5);
        print_ui(&preedit, &suggestions, &engine);

        let mut input = String::new();
        stdin().read_line(&mut input).unwrap();
        let cmd = input.trim();

        match cmd {
            "exit" => break,
            "" => { // Enter key - commit the top suggestion or transliteration
                let to_commit = if !suggestions.is_empty() {
                    suggestions[0].0.clone()
                } else {
                    engine.romanizer.transliterate(&preedit)
                };
                println!("\nCommitting: '{}'", to_commit);
                engine.user_confirms(&preedit, &to_commit);
                preedit.clear();
            }
            s if s.starts_with(':') && s.len() > 1 => { // Select suggestion :1, :2 etc
                if let Ok(n) = s[1..].parse::<usize>() {
                    if n > 0 && n <= suggestions.len() {
                        let chosen = suggestions[n - 1].0.clone();
                        println!("\nCommitting: '{}'", chosen);
                        engine.user_confirms(&preedit, &chosen);
                        preedit.clear();
                    }
                }
            }
            s => { // Append to preedit
                preedit.push_str(s);
            }
        }
    }

    println!("\nSaving dictionary...");
    if let Err(e) = engine.save_dictionary() {
        eprintln!("[ERROR] Could not save dictionary: {}", e);
    } else {
        println!("Dictionary saved to '{}'", DICTIONARY_PATH);
    }
}

fn print_ui(preedit: &str, suggestions: &[(String, u64)], engine: &ImeEngine) {
    // Basic clear screen for simplicity
    print!("\x1B[2J\x1B[1;1H");
    println!("Nepali Smart IME Simulator (Full Version)");
    println!("---------------------------------------------------------------");
    println!("Type and press [Enter] to commit, or type a letter to continue.");
    println!("Select with ':1', ':2'. 'exit' to save and quit.\n");

    println!("Context: {:?}", engine.context_model);

    println!("\nPre-edit: [{}]", preedit);
    println!("Transliteration -> {}", engine.romanizer.transliterate(preedit));

    if !suggestions.is_empty() {
        println!("\nSuggestions (re-ranked by context):");
        for (i, (word, score)) in suggestions.iter().enumerate() {
            println!("  :{}: {} (score: {})", i + 1, word, score);
        }
    } else {
        println!("\nNo suggestions found.");
    }
    print!("\n> ");
    stdout().flush().unwrap();
}