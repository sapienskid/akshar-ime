// No longer need HashMap
// use std::collections::HashMap; 

const HALANTA: char = '\u{094d}';

/// A stateful Romanization to Devanagari converter.
pub struct RomanizationEngine; // No longer needs fields

impl RomanizationEngine {
    pub fn new() -> Self {
        // No longer needs to initialize anything
        Self
    }

    // `is_nepali_consonant` function removed as it was unused.

    /// Transliterates a full roman string.
    pub fn transliterate(&self, roman: &str) -> String {
        let mut result = String::new();
        let mut chars = roman.chars().peekable();
        let mut last_was_consonant = false;

        while let Some(c) = chars.next() {
            match c {
                'a' => {
                    if last_was_consonant {
                        if result.ends_with(HALANTA) {
                            result.pop();
                        }
                    } else {
                        result.push('अ');
                    }
                    last_was_consonant = false;
                },
                'i' => result.push(if last_was_consonant {'ि'} else {'इ'}),
                'u' => result.push(if last_was_consonant {'ु'} else {'उ'}),
                
                'k' | 'c' | 't' | 'p' | 'b' | 'm' | 'y' | 'r' | 'l' | 'v' | 's' | 'h' | 'g' | 'd' | 'n' => {
                    let mut cons = c.to_string();
                    if let Some(&next_c) = chars.peek() {
                        if next_c == 'h' {
                           cons.push(next_c);
                           chars.next(); 
                        }
                    }

                    if last_was_consonant && result.ends_with(HALANTA) {
                        result.pop();
                    }
                    
                    if let Some(nep_c) = self.get_consonant(&cons) {
                        result.push(nep_c);
                        result.push(HALANTA); 
                        last_was_consonant = true;
                    } else {
                         result.push(c);
                         last_was_consonant = false;
                    }
                }
                _ => { 
                    if result.ends_with(HALANTA) { result.pop(); }
                    result.push(c);
                    last_was_consonant = false;
                }
            }
        }
        
        if result.ends_with(HALANTA) {
            result.pop();
        }

        result
    }

    fn get_consonant(&self, s: &str) -> Option<char> {
        match s {
            "k" => Some('क'), "kh" => Some('ख'), "g" => Some('ग'),
            "gh" => Some('घ'), "n" => Some('न'), "c" => Some('च'),
            "ch" => Some('छ'), "j" => Some('ज'), "jh" => Some('झ'),
            "t" => Some('त'), "th" => Some('थ'), "d" => Some('द'),
            "dh" => Some('ध'), "p" => Some('प'), "ph" => Some('फ'),
            "b" => Some('ब'), "bh" => Some('भ'), "m" => Some('म'),
            "y" => Some('य'), "r" => Some('र'), "l" => Some('ल'),
            "v" => Some('व'), "s" => Some('स'), "sh" => Some('श'),
            "h" => Some('ह'),
            _ => None,
        }
    }
}