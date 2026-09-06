// Diagnostic: what score does each candidate source actually contribute?
use akshar_ime::ImeEngine;

fn main() {
    let mut e = ImeEngine::new();
    e.user_confirms("kathmandu", "काठमाडौँ");
    for q in ["kathmandu", "kathmandau", "kathmndu"] {
        println!("\n=== query {q:?} ===");
        for (i, (s, sc)) in e.get_suggestions(q, 8).into_iter().enumerate() {
            let tag = if s == "काठमाडौँ" {
                "  <-- LEARNED"
            } else {
                ""
            };
            println!("  {i:2}. {sc:>9}  {s}{tag}");
        }
    }
    println!("\nBands: decoder top-1 = 800000/(1+cost), user trie = 900000+freq,");
    println!("       user fuzzy = 50000 - 12000*distance, corpus-only lexicon = 5000");
}
