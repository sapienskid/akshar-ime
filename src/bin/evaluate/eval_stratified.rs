//! Stratified, Homophone-Aware & Contextual Nepali IME Benchmark Runner.

use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EvalStratum {
    CoreNative,
    AgglutinativeInflected,
    ColloquialSocialMedia,
    CuratedEnglishLoanword,
    OovEnglishLoanword,
    HomographCollisionGuard,
    TimeUnitsAndAcronyms,
}

#[derive(Debug, Clone)]
pub struct MultiRefTestCase {
    pub roman: String,
    pub primary_gold: String,
    pub valid_homophones: Vec<String>,
    pub stratum: EvalStratum,
}

#[derive(Debug, Default, Clone)]
pub struct StratumMetrics {
    pub total: usize,
    pub strict_primary_top1: usize,
    pub any_valid_top1: usize,
    pub in_list_at_5: usize,
    pub in_list_at_8: usize,
    pub reachable_at_50: usize,
    pub homophone_cluster_recall_at_5_hits: usize,
    pub homophone_cluster_recall_at_5_total: usize,
    pub mrr_sum: f64,
    pub total_keystrokes_needed: usize,
    pub min_possible_keystrokes: usize,
}

impl StratumMetrics {
    pub fn record(&mut self, tc: &MultiRefTestCase, candidates: &[String]) {
        self.total += 1;
        self.min_possible_keystrokes += tc.roman.len() + 1;

        let primary_pos = candidates.iter().position(|c| c == &tc.primary_gold);
        let any_valid_pos = candidates
            .iter()
            .position(|c| c == &tc.primary_gold || tc.valid_homophones.contains(c));

        if primary_pos == Some(0) {
            self.strict_primary_top1 += 1;
        }
        if any_valid_pos == Some(0) {
            self.any_valid_top1 += 1;
        }

        if let Some(pos) = any_valid_pos {
            if pos < 5 {
                self.in_list_at_5 += 1;
            }
            if pos < 8 {
                self.in_list_at_8 += 1;
            }
            if pos < 50 {
                self.reachable_at_50 += 1;
                self.mrr_sum += 1.0 / ((pos + 1) as f64);
            }
            let commit_keys = match pos {
                0 => 1,
                1..=4 => 2,
                5..=7 => 3,
                _ => 6,
            };
            self.total_keystrokes_needed += tc.roman.len() + commit_keys;
        } else {
            self.total_keystrokes_needed += tc.roman.len() + 8;
        }

        if !tc.valid_homophones.is_empty() {
            let top5 = &candidates[..candidates.len().min(5)];
            for h in &tc.valid_homophones {
                self.homophone_cluster_recall_at_5_total += 1;
                if top5.contains(h) {
                    self.homophone_cluster_recall_at_5_hits += 1;
                }
            }
        }
    }

    pub fn summary_line(&self, label: &str) -> String {
        if self.total == 0 {
            return format!("{:<26} | N=0", label);
        }
        let n = self.total as f64;
        let strict_t1 = 100.0 * (self.strict_primary_top1 as f64) / n;
        let valid_t1 = 100.0 * (self.any_valid_top1 as f64) / n;
        let top5 = 100.0 * (self.in_list_at_5 as f64) / n;
        let top8 = 100.0 * (self.in_list_at_8 as f64) / n;
        let reach50 = 100.0 * (self.reachable_at_50 as f64) / n;
        let mrr = self.mrr_sum / n;
        let ksr =
            (self.total_keystrokes_needed as f64) / (self.min_possible_keystrokes.max(1) as f64);
        format!(
            "{:<26} | N={:<4} | Valid-Top1: {:>5.2}% (Strict: {:>5.2}%) | Popup@5: {:>5.2}% | InList@8: {:>5.2}% | Reach@50: {:>5.2}% | MRR: {:.3} | KeystrokeRatio: {:.2}x",
            label, self.total, valid_t1, strict_t1, top5, top8, reach50, mrr, ksr
        )
    }
}

pub fn run_stratified_evaluation<F>(
    cases: &[MultiRefTestCase],
    mut query_fn: F,
) -> HashMap<EvalStratum, StratumMetrics>
where
    F: FnMut(&str, usize) -> Vec<String>,
{
    let mut by_stratum: HashMap<EvalStratum, StratumMetrics> = HashMap::new();
    let mut overall = StratumMetrics::default();

    for tc in cases {
        let candidates = query_fn(&tc.roman, 50);
        overall.record(tc, &candidates);
        by_stratum
            .entry(tc.stratum)
            .or_default()
            .record(tc, &candidates);
    }

    println!("{}", overall.summary_line("OVERALL (ALL STRATA)"));
    for (stratum, label) in [
        (EvalStratum::CoreNative, "1. Core Native Nepali"),
        (
            EvalStratum::AgglutinativeInflected,
            "2. Agglutinative Inflected",
        ),
        (
            EvalStratum::ColloquialSocialMedia,
            "3. Colloquial Social Media",
        ),
        (
            EvalStratum::CuratedEnglishLoanword,
            "4a. Curated English Loanwords",
        ),
        (
            EvalStratum::OovEnglishLoanword,
            "4b. Unseen/OOV English (General)",
        ),
        (
            EvalStratum::HomographCollisionGuard,
            "5. Homograph Collision Guard",
        ),
        (
            EvalStratum::TimeUnitsAndAcronyms,
            "6. Time, Units & Acronyms",
        ),
    ] {
        if let Some(m) = by_stratum.get(&stratum) {
            println!("{}", m.summary_line(label));
        }
    }

    by_stratum
}

pub fn build_curated_benchmark_cases() -> Vec<MultiRefTestCase> {
    let raw: &[(&str, &str, &[&str], EvalStratum)] = &[
        // Stratum 1: Core Native Nepali
        ("namaste", "नमस्ते", &["नमस्ते"], EvalStratum::CoreNative),
        ("nepal", "नेपाल", &[], EvalStratum::CoreNative),
        (
            "kathmandu",
            "काठमाडौँ",
            &["काठमाडौं"],
            EvalStratum::CoreNative,
        ),
        ("dhanyabad", "धन्यवाद", &[], EvalStratum::CoreNative),
        ("sathi", "साथी", &[], EvalStratum::CoreNative),
        ("saathi", "साथी", &[], EvalStratum::CoreNative),
        ("pani", "पनि", &["पानी"], EvalStratum::CoreNative),
        ("paani", "पानी", &[], EvalStratum::CoreNative),
        ("khana", "खाना", &[], EvalStratum::CoreNative),
        ("ramro", "राम्रो", &[], EvalStratum::CoreNative),
        ("bhaneko", "भनेको", &[], EvalStratum::CoreNative),
        ("thik", "ठीक", &["ठिक"], EvalStratum::CoreNative),
        ("theek", "ठीक", &["ठिक"], EvalStratum::CoreNative),
        ("maya", "माया", &[], EvalStratum::CoreNative),
        ("samaya", "समय", &[], EvalStratum::CoreNative),
        ("desh", "देश", &[], EvalStratum::CoreNative),
        ("bidesh", "विदेश", &[], EvalStratum::CoreNative),
        ("pusta", "पुस्ता", &[], EvalStratum::CoreNative),
        ("manna", "मान्न", &["मन्न"], EvalStratum::CoreNative),
        ("chahana", "चाहना", &[], EvalStratum::CoreNative),
        ("prashna", "प्रश्न", &[], EvalStratum::CoreNative),
        ("uttar", "उत्तर", &[], EvalStratum::CoreNative),
        ("ra", "र", &[], EvalStratum::CoreNative),
        ("ani", "अनि", &[], EvalStratum::CoreNative),
        ("kina", "किन", &[], EvalStratum::CoreNative),
        ("kahile", "कहिले", &[], EvalStratum::CoreNative),
        (
            "tapaiko",
            "तपाईंको",
            &["तपाइको", "तपाईको"],
            EvalStratum::CoreNative,
        ),
        ("tapai", "तपाईं", &["तपाइँ", "तपाइ"], EvalStratum::CoreNative),
        ("hami", "हामी", &["हामि"], EvalStratum::CoreNative),
        ("timi", "तिमी", &["तिमि"], EvalStratum::CoreNative),
        ("mero", "मेरो", &[], EvalStratum::CoreNative),
        // Stratum 2: Agglutinative Inflected
        (
            "gharmathi",
            "घरमाथि",
            &[],
            EvalStratum::AgglutinativeInflected,
        ),
        (
            "gharbata",
            "घरबाट",
            &[],
            EvalStratum::AgglutinativeInflected,
        ),
        (
            "saharma",
            "सहरमा",
            &["शहरमा"],
            EvalStratum::AgglutinativeInflected,
        ),
        ("deshko", "देशको", &[], EvalStratum::AgglutinativeInflected),
        (
            "deshharu",
            "देशहरू",
            &["देशहरु"],
            EvalStratum::AgglutinativeInflected,
        ),
        (
            "manchhele",
            "मान्छेले",
            &[],
            EvalStratum::AgglutinativeInflected,
        ),
        (
            "sathisanga",
            "साथीसँग",
            &["साथिसँग"],
            EvalStratum::AgglutinativeInflected,
        ),
        ("bhanda", "भन्दा", &[], EvalStratum::AgglutinativeInflected),
        (
            "nepaldekhi",
            "नेपालदेखि",
            &[],
            EvalStratum::AgglutinativeInflected,
        ),
        (
            "gharharuma",
            "घरहरूमा",
            &["घरहरुमा"],
            EvalStratum::AgglutinativeInflected,
        ),
        (
            "ketiharulai",
            "केटीहरूलाई",
            &["केटीहरुलाई", "केटिहरुलाई"],
            EvalStratum::AgglutinativeInflected,
        ),
        (
            "facebookma",
            "फेसबुकमा",
            &[],
            EvalStratum::AgglutinativeInflected,
        ),
        (
            "officebata",
            "अफिसबाट",
            &["कार्यालयबाट"],
            EvalStratum::AgglutinativeInflected,
        ),
        (
            "schoolma",
            "स्कुलमा",
            &["स्कूलमा"],
            EvalStratum::AgglutinativeInflected,
        ),
        (
            "linkharulai",
            "लिङ्कहरूलाई",
            &["लिंकहरूलाई", "लिंकहरुलाई", "लिङ्कहरुलाई"],
            EvalStratum::AgglutinativeInflected,
        ),
        (
            "videoma",
            "भिडियोमा",
            &[],
            EvalStratum::AgglutinativeInflected,
        ),
        (
            "hospitaldekhi",
            "अस्पतालदेखि",
            &["हस्पिटलदेखि"],
            EvalStratum::AgglutinativeInflected,
        ),
        (
            "computerko",
            "कम्प्युटरको",
            &[],
            EvalStratum::AgglutinativeInflected,
        ),
        (
            "collegema",
            "कलेजमा",
            &[],
            EvalStratum::AgglutinativeInflected,
        ),
        (
            "ticketbata",
            "टिकटबाट",
            &[],
            EvalStratum::AgglutinativeInflected,
        ),
        // Stratum 3: Colloquial Social Media
        ("xa", "छ", &[], EvalStratum::ColloquialSocialMedia),
        ("xan", "छन्", &[], EvalStratum::ColloquialSocialMedia),
        ("xau", "छौ", &[], EvalStratum::ColloquialSocialMedia),
        ("xu", "छु", &[], EvalStratum::ColloquialSocialMedia),
        ("vayo", "भयो", &[], EvalStratum::ColloquialSocialMedia),
        ("bhayo", "भयो", &[], EvalStratum::ColloquialSocialMedia),
        ("vo", "भो", &[], EvalStratum::ColloquialSocialMedia),
        (
            "hudaina",
            "हुँदैन",
            &["हुदैन"],
            EvalStratum::ColloquialSocialMedia,
        ),
        ("gardai", "गर्दै", &[], EvalStratum::ColloquialSocialMedia),
        ("gardaina", "गर्दैन", &[], EvalStratum::ColloquialSocialMedia),
        ("hunchha", "हुन्छ", &[], EvalStratum::ColloquialSocialMedia),
        ("hunxa", "हुन्छ", &[], EvalStratum::ColloquialSocialMedia),
        (
            "bhanum",
            "भनुम्",
            &["भनूँ", "भनुम"],
            EvalStratum::ColloquialSocialMedia,
        ),
        ("fatafat", "फटाफट", &[], EvalStratum::ColloquialSocialMedia),
        ("jhuto", "झुटो", &[], EvalStratum::ColloquialSocialMedia),
        ("kasto", "कस्तो", &[], EvalStratum::ColloquialSocialMedia),
        ("sabbai", "सबै", &[], EvalStratum::ColloquialSocialMedia),
        ("yesto", "यस्तो", &[], EvalStratum::ColloquialSocialMedia),
        ("tyasto", "त्यस्तो", &[], EvalStratum::ColloquialSocialMedia),
        ("khubai", "खुबै", &[], EvalStratum::ColloquialSocialMedia),
        // Stratum 4a: Curated English Loanwords & Code-Mixed (in dictionary)
        (
            "biscuit",
            "बिस्कुट",
            &["बिस्किट"],
            EvalStratum::CuratedEnglishLoanword,
        ),
        (
            "hospital",
            "अस्पताल",
            &["हस्पिटल"],
            EvalStratum::CuratedEnglishLoanword,
        ),
        (
            "doctor",
            "डाक्टर",
            &["डक्टर"],
            EvalStratum::CuratedEnglishLoanword,
        ),
        (
            "police",
            "पुलिस",
            &["प्रहरी"],
            EvalStratum::CuratedEnglishLoanword,
        ),
        (
            "computer",
            "कम्प्युटर",
            &[],
            EvalStratum::CuratedEnglishLoanword,
        ),
        ("college", "कलेज", &[], EvalStratum::CuratedEnglishLoanword),
        ("campus", "क्याम्पस", &[], EvalStratum::CuratedEnglishLoanword),
        ("mobile", "मोबाइल", &[], EvalStratum::CuratedEnglishLoanword),
        ("phone", "फोन", &[], EvalStratum::CuratedEnglishLoanword),
        ("ticket", "टिकट", &[], EvalStratum::CuratedEnglishLoanword),
        ("facebook", "फेसबुक", &[], EvalStratum::CuratedEnglishLoanword),
        ("youtube", "युट्युब", &[], EvalStratum::CuratedEnglishLoanword),
        (
            "internet",
            "इन्टरनेट",
            &[],
            EvalStratum::CuratedEnglishLoanword,
        ),
        (
            "bank",
            "बैंक",
            &["ब्याङ्क"],
            EvalStratum::CuratedEnglishLoanword,
        ),
        (
            "hotel",
            "होटल",
            &["होटेल"],
            EvalStratum::CuratedEnglishLoanword,
        ),
        ("bus", "बस", &[], EvalStratum::CuratedEnglishLoanword),
        (
            "station",
            "स्टेसन",
            &["स्टेशन"],
            EvalStratum::CuratedEnglishLoanword,
        ),
        ("driver", "ड्राइभर", &[], EvalStratum::CuratedEnglishLoanword),
        ("report", "रिपोर्ट", &[], EvalStratum::CuratedEnglishLoanword),
        (
            "meeting",
            "मिटिङ",
            &["मिटिङ्ग"],
            EvalStratum::CuratedEnglishLoanword,
        ),
        ("science", "साइन्स", &[], EvalStratum::CuratedEnglishLoanword),
        (
            "message",
            "म्यासेज",
            &["मेसेज"],
            EvalStratum::CuratedEnglishLoanword,
        ),
        (
            "link",
            "लिङ्क",
            &["लिंक"],
            EvalStratum::CuratedEnglishLoanword,
        ),
        (
            "download",
            "डाउनलोड",
            &[],
            EvalStratum::CuratedEnglishLoanword,
        ),
        ("system", "सिस्टम", &[], EvalStratum::CuratedEnglishLoanword),
        (
            "office",
            "अफिस",
            &["कार्यालय"],
            EvalStratum::CuratedEnglishLoanword,
        ),
        (
            "email",
            "इमेल",
            &["ईमेल"],
            EvalStratum::CuratedEnglishLoanword,
        ),
        ("camera", "क्यामेरा", &[], EvalStratum::CuratedEnglishLoanword),
        ("radio", "रेडियो", &[], EvalStratum::CuratedEnglishLoanword),
        ("video", "भिडियो", &[], EvalStratum::CuratedEnglishLoanword),
        // Stratum 4b: Unseen / Out-of-Vocabulary (OOV) English Loanwords (Generalization)
        (
            "refrigerator",
            "रेफ्रिजरेटर",
            &["रेफ्रिजेरेटर"],
            EvalStratum::OovEnglishLoanword,
        ),
        (
            "calculator",
            "क्याल्कुलेटर",
            &["काल्कुलेटर", "कालकुलेटर"],
            EvalStratum::OovEnglishLoanword,
        ),
        (
            "calendar",
            "क्यालेन्डर",
            &["क्यालेण्डर", "कालेन्डर"],
            EvalStratum::OovEnglishLoanword,
        ),
        (
            "ambulance",
            "एम्बुलेन्स",
            &["अम्बुलेन्स"],
            EvalStratum::OovEnglishLoanword,
        ),
        (
            "helicopter",
            "हेलिकप्टर",
            &["हेलिकोप्टर"],
            EvalStratum::OovEnglishLoanword,
        ),
        (
            "keyboard",
            "किबोर्ड",
            &["कीबोर्ड"],
            EvalStratum::OovEnglishLoanword,
        ),
        ("charger", "चार्जर", &[], EvalStratum::OovEnglishLoanword),
        ("display", "डिस्प्ले", &[], EvalStratum::OovEnglishLoanword),
        (
            "monitor",
            "मनिटर",
            &["मोनिटर"],
            EvalStratum::OovEnglishLoanword,
        ),
        ("headphone", "हेडफोन", &[], EvalStratum::OovEnglishLoanword),
        ("projector", "प्रोजेक्टर", &[], EvalStratum::OovEnglishLoanword),
        (
            "microphone",
            "माइक्रोफोन",
            &[],
            EvalStratum::OovEnglishLoanword,
        ),
        (
            "battery",
            "ब्याट्री",
            &["ब्याट्री"],
            EvalStratum::OovEnglishLoanword,
        ),
        ("printer", "प्रिन्टर", &[], EvalStratum::OovEnglishLoanword),
        ("scanner", "स्क्यानर", &[], EvalStratum::OovEnglishLoanword),
        ("laptop", "ल्यापटप", &[], EvalStratum::OovEnglishLoanword),
        ("tablet", "ट्याब्लेट", &[], EvalStratum::OovEnglishLoanword),
        ("browser", "ब्राउजर", &[], EvalStratum::OovEnglishLoanword),
        ("website", "वेबसाइट", &[], EvalStratum::OovEnglishLoanword),
        ("server", "सर्भर", &[], EvalStratum::OovEnglishLoanword),
        (
            "software",
            "सफ्टवेयर",
            &["सोफ्टवेयर", "सोफटवेयर"],
            EvalStratum::OovEnglishLoanword,
        ),
        ("hardware", "हार्डवेयर", &[], EvalStratum::OovEnglishLoanword),
        ("network", "नेटवर्क", &[], EvalStratum::OovEnglishLoanword),
        ("password", "पासवर्ड", &[], EvalStratum::OovEnglishLoanword),
        ("account", "एकाउन्ट", &[], EvalStratum::OovEnglishLoanword),
        ("profile", "प्रोफाइल", &[], EvalStratum::OovEnglishLoanword),
        (
            "notification",
            "नोटिफिकेसन",
            &["नोटिफिकेशन"],
            EvalStratum::OovEnglishLoanword,
        ),
        (
            "version",
            "भर्सन",
            &["भर्जन"],
            EvalStratum::OovEnglishLoanword,
        ),
        ("payment", "पेमेन्ट", &[], EvalStratum::OovEnglishLoanword),
        (
            "application",
            "एप्लिकेसन",
            &["एप्लिकेशन"],
            EvalStratum::OovEnglishLoanword,
        ),
        // Stratum 5: Homograph Collision Guard
        ("man", "मन", &["म्यान"], EvalStratum::HomographCollisionGuard),
        ("din", "दिन", &["डिन"], EvalStratum::HomographCollisionGuard),
        (
            "ram",
            "राम",
            &["र्याम"],
            EvalStratum::HomographCollisionGuard,
        ),
        (
            "pan",
            "पान",
            &["पात", "प्यान"],
            EvalStratum::HomographCollisionGuard,
        ),
        (
            "kam",
            "काम",
            &["कम", "क्याम"],
            EvalStratum::HomographCollisionGuard,
        ),
        ("to", "तो", &[], EvalStratum::HomographCollisionGuard),
        ("ho", "हो", &[], EvalStratum::HomographCollisionGuard),
        ("so", "सो", &[], EvalStratum::HomographCollisionGuard),
        ("ko", "को", &[], EvalStratum::HomographCollisionGuard),
        ("le", "ले", &[], EvalStratum::HomographCollisionGuard),
        ("ma", "म", &[], EvalStratum::HomographCollisionGuard),
        ("ta", "त", &[], EvalStratum::HomographCollisionGuard),
        ("ti", "ती", &[], EvalStratum::HomographCollisionGuard),
        (
            "am",
            "आम",
            &["एम", "AM"],
            EvalStratum::HomographCollisionGuard,
        ),
        ("das", "दस", &["दश"], EvalStratum::HomographCollisionGuard),
        ("bas", "बस", &[], EvalStratum::HomographCollisionGuard),
        ("sat", "सात", &[], EvalStratum::HomographCollisionGuard),
        ("hat", "हात", &[], EvalStratum::HomographCollisionGuard),
        ("rat", "रात", &[], EvalStratum::HomographCollisionGuard),
        ("bat", "बात", &["बाट"], EvalStratum::HomographCollisionGuard),
        ("pat", "पात", &[], EvalStratum::HomographCollisionGuard),
        ("mat", "मत", &[], EvalStratum::HomographCollisionGuard),
        ("mal", "माल", &[], EvalStratum::HomographCollisionGuard),
        ("sal", "साल", &[], EvalStratum::HomographCollisionGuard),
        ("hal", "हाल", &[], EvalStratum::HomographCollisionGuard),
        ("pal", "पाल", &[], EvalStratum::HomographCollisionGuard),
        ("tal", "ताल", &[], EvalStratum::HomographCollisionGuard),
        // Stratum 6: Time Expressions, Measurement Units & Latin Acronyms
        (
            "10am",
            "१० AM",
            &["१० am", "१० एएम", "10 AM"],
            EvalStratum::TimeUnitsAndAcronyms,
        ),
        (
            "10pm",
            "१० PM",
            &["१० pm", "१० पीएम", "10 PM"],
            EvalStratum::TimeUnitsAndAcronyms,
        ),
        (
            "10:30am",
            "१०:३० AM",
            &["१०:३० am", "१०:३० एएम", "10:30 AM"],
            EvalStratum::TimeUnitsAndAcronyms,
        ),
        (
            "10:30pm",
            "१०:३० PM",
            &["१०:३० pm", "१०:३० पीएम", "10:30 PM"],
            EvalStratum::TimeUnitsAndAcronyms,
        ),
        (
            "a.m.",
            "AM",
            &["a.m.", "एएम"],
            EvalStratum::TimeUnitsAndAcronyms,
        ),
        (
            "p.m.",
            "PM",
            &["p.m.", "पीएम", "पिएम"],
            EvalStratum::TimeUnitsAndAcronyms,
        ),
        (
            "AM",
            "AM",
            &["एएम", "आम"],
            EvalStratum::TimeUnitsAndAcronyms,
        ),
        (
            "PM",
            "PM",
            &["पीएम", "पिएम"],
            EvalStratum::TimeUnitsAndAcronyms,
        ),
        (
            "pm",
            "पीएम",
            &["पिएम", "PM"],
            EvalStratum::TimeUnitsAndAcronyms,
        ),
        (
            "10kg",
            "१० kg",
            &["१० केजी", "10 kg"],
            EvalStratum::TimeUnitsAndAcronyms,
        ),
        (
            "10km",
            "१० km",
            &["१० किमी", "10 km"],
            EvalStratum::TimeUnitsAndAcronyms,
        ),
        (
            "10mb",
            "१० MB",
            &["१० mb", "10 MB"],
            EvalStratum::TimeUnitsAndAcronyms,
        ),
        ("OTP", "OTP", &[], EvalStratum::TimeUnitsAndAcronyms),
        ("PDF", "PDF", &[], EvalStratum::TimeUnitsAndAcronyms),
        ("OK", "OK", &["ओके"], EvalStratum::TimeUnitsAndAcronyms),
    ];

    raw.iter()
        .map(|(roman, gold, homos, stratum)| MultiRefTestCase {
            roman: (*roman).to_string(),
            primary_gold: (*gold).to_string(),
            valid_homophones: homos.iter().map(|s| (*s).to_string()).collect(),
            stratum: *stratum,
        })
        .collect()
}

fn load_engine(model_path: Option<&std::path::Path>) -> akshar_ime::ImeEngine {
    if let Some(p) = model_path {
        if p.exists() {
            if let Ok(eng) = akshar_ime::ImeEngine::from_unified_file(p) {
                return eng;
            }
        }
    }
    akshar_ime::ImeEngine::new()
}

fn run_interactive(engine: akshar_ime::ImeEngine) {
    use std::io::{stdin, stdout, Write};
    println!("=== Akshar IME Interactive Harness ===");
    println!("Type a romanized query and press Enter. Type 'quit' or Ctrl-C to exit.");
    print!("> ");
    let _ = stdout().flush();

    let mut line = String::new();
    while stdin().read_line(&mut line).is_ok() {
        let input = line.trim();
        if input.is_empty() {
            print!("> ");
            let _ = stdout().flush();
            line.clear();
            continue;
        }
        if input == "quit" || input == "exit" {
            break;
        }

        let suggestions = engine.get_suggestions(input, 8);
        println!(
            "Input: \"{}\" -> Suggestions (k={}):",
            input,
            suggestions.len()
        );
        for (idx, (word, score)) in suggestions.iter().enumerate() {
            println!("  [{}] {:<16} (score: {})", idx + 1, word, score);
        }

        line.clear();
        print!("> ");
        let _ = stdout().flush();
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model_path: Option<std::path::PathBuf> = None;
    let mut interactive = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--model" => {
                if let Some(m) = args.next() {
                    model_path = Some(std::path::PathBuf::from(m));
                }
            }
            "--interactive" | "-i" => {
                interactive = true;
            }
            "--help" | "-h" => {
                println!("Usage: eval_stratified [--model <path>] [--interactive]");
                return Ok(());
            }
            _ => {}
        }
    }

    if model_path.is_none() && std::path::Path::new("data/akshar.model").exists() {
        model_path = Some(std::path::PathBuf::from("data/akshar.model"));
    }

    let engine = load_engine(model_path.as_deref());

    if interactive {
        run_interactive(engine);
        return Ok(());
    }

    let cases = build_curated_benchmark_cases();
    println!("=== Akshar IME Stratified Multi-Strata Benchmark ===");
    println!("Total test cases: {}", cases.len());

    let _metrics = run_stratified_evaluation(&cases, |roman, count| {
        engine
            .get_suggestions(roman, count)
            .into_iter()
            .map(|(w, _)| w)
            .collect()
    });

    Ok(())
}
