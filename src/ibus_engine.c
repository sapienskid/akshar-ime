#include <ibus.h>
#include <jansson.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

static void log_message(const char *format, ...) {
    FILE *log_file = fopen("/run/media/sapiens/Development/Development/nepali-ime/target/c_engine.log", "a");
    if (log_file) {
        va_list args;
        va_start(args, format);
        vfprintf(log_file, format, args);
        va_end(args);
        fprintf(log_file, "\n");
        fclose(log_file);
    }
}

// --- Rust FFI function declarations ---
void nepali_ime_engine_init(void);
void nepali_ime_engine_destroy(void);
char* nepali_ime_get_suggestions(const char* prefix);
void nepali_ime_confirm_word(const char* roman, const char* nepali);
void nepali_ime_free_string(char* s);

// --- GObject Boilerplate ---
typedef struct _IBusNepaliEngine IBusNepaliEngine;
typedef struct _IBusNepaliEngineClass IBusNepaliEngineClass;

struct _IBusNepaliEngine {
    IBusEngine parent;
    IBusLookupTable *table;
    GString *preedit_string;
};

struct _IBusNepaliEngineClass { IBusEngineClass parent; };

static guint g_engine_instance_count = 0;

static void ibus_nepali_engine_class_init(IBusNepaliEngineClass *klass);
static void ibus_nepali_engine_init_instance(IBusNepaliEngine *engine);
static void ibus_nepali_engine_finalize(GObject *object);
static gboolean ibus_nepali_engine_process_key_event(IBusEngine *engine, guint keyval, guint keycode, guint modifiers);
static void ibus_nepali_engine_candidate_clicked(IBusEngine *engine, guint index, guint button, guint state);

G_DEFINE_TYPE(IBusNepaliEngine, ibus_nepali_engine, IBUS_TYPE_ENGINE)
#define IBUS_TYPE_NEPALI_ENGINE (ibus_nepali_engine_get_type())

static void update_preedit_and_lookup(IBusNepaliEngine *nepali_engine);
static void commit_selection(IBusNepaliEngine *nepali_engine, const gchar *commit_str);

static void ibus_nepali_engine_class_init(IBusNepaliEngineClass *klass) {
    log_message("[C LOG] ibus_nepali_engine_class_init: Initializing class.");
    IBusEngineClass *engine_class = IBUS_ENGINE_CLASS(klass);
    GObjectClass *object_class = G_OBJECT_CLASS(klass);

    engine_class->process_key_event = ibus_nepali_engine_process_key_event;
    engine_class->candidate_clicked = ibus_nepali_engine_candidate_clicked;
    object_class->finalize = ibus_nepali_engine_finalize;
}

static void ibus_nepali_engine_init_instance(IBusNepaliEngine *engine) {
    log_message("[C LOG] ibus_nepali_engine_init_instance: Creating new engine instance.");
    engine->preedit_string = g_string_new("");
    engine->table = ibus_lookup_table_new(10, 0, TRUE, TRUE);
    g_object_ref_sink(engine->table);

    if (g_engine_instance_count == 0) {
        log_message("[C LOG] First instance, initializing Rust engine.");
        nepali_ime_engine_init();
    }
    g_engine_instance_count++;
}

static void ibus_nepali_engine_init(IBusNepaliEngine *engine) {
    ibus_nepali_engine_init_instance(engine);
}

static void ibus_nepali_engine_finalize(GObject *object) {
    log_message("[C LOG] ibus_nepali_engine_finalize: Finalizing engine instance.");
    IBusNepaliEngine *engine = (IBusNepaliEngine *)object;
    g_string_free(engine->preedit_string, TRUE);
    g_object_unref(engine->table);

    g_engine_instance_count--;
    if (g_engine_instance_count == 0) {
        log_message("[C LOG] Last instance, destroying Rust engine.");
        nepali_ime_engine_destroy();
    }
    
    G_OBJECT_CLASS(ibus_nepali_engine_parent_class)->finalize(object);
}

static void clear_preedit(IBusNepaliEngine *nepali_engine) {
    g_string_set_size(nepali_engine->preedit_string, 0);
    ibus_engine_hide_preedit_text((IBusEngine *)nepali_engine);
    ibus_engine_hide_lookup_table((IBusEngine *)nepali_engine);
}

static void update_preedit_and_lookup(IBusNepaliEngine *nepali_engine) {
    IBusEngine *engine = (IBusEngine *)nepali_engine;
    const char* preedit_str = nepali_engine->preedit_string->str;
    log_message("[C LOG] update_preedit_and_lookup called with: '%s'", preedit_str);

    if (strlen(preedit_str) == 0) {
        clear_preedit(nepali_engine);
        return;
    }

    IBusText *preedit_text = ibus_text_new_from_string(preedit_str);
    ibus_engine_update_preedit_text(engine, preedit_text, strlen(preedit_str), TRUE);

    ibus_lookup_table_clear(nepali_engine->table);
    char* suggestions_json = nepali_ime_get_suggestions(preedit_str);
    log_message("[C LOG] Got suggestions from Rust: %s", suggestions_json);
    
    json_error_t error;
    json_t *root = json_loads(suggestions_json, 0, &error);
    
    if (root && json_is_array(root)) {
        size_t i;
        json_t *value;
        json_array_foreach(root, i, value) {
            if (json_is_string(value)) {
                IBusText *candidate_text = ibus_text_new_from_string(json_string_value(value));
                ibus_lookup_table_append_candidate(nepali_engine->table, candidate_text);
            }
        }
        json_decref(root);
    } else {
        log_message("[C LOG] Failed to parse JSON suggestions. Error: %s", error.text);
    }
    
    nepali_ime_free_string(suggestions_json);

    if (ibus_lookup_table_get_number_of_candidates(nepali_engine->table) > 0) {
        ibus_engine_update_lookup_table(engine, nepali_engine->table, TRUE);
    } else {
        ibus_engine_hide_lookup_table(engine);
    }
}

static void commit_selection(IBusNepaliEngine *nepali_engine, const gchar *commit_str) {
    if (commit_str == NULL || nepali_engine->preedit_string->len == 0) return;
    
    log_message("[C LOG] Committing text: '%s' for roman: '%s'", commit_str, nepali_engine->preedit_string->str);
    IBusEngine *engine = (IBusEngine *)nepali_engine;
    IBusText *text = ibus_text_new_from_string(commit_str);
    
    ibus_engine_commit_text(engine, text);
    nepali_ime_confirm_word(nepali_engine->preedit_string->str, commit_str);
    
    clear_preedit(nepali_engine);
}

static void commit_candidate_by_index(IBusNepaliEngine *nepali_engine, guint index) {
    IBusText *text = ibus_lookup_table_get_candidate(nepali_engine->table, index);
    if (text) {
        commit_selection(nepali_engine, text->text);
    }
}

static gboolean ibus_nepali_engine_process_key_event(IBusEngine *engine, guint keyval, guint keycode, guint modifiers) {
    IBusNepaliEngine *nepali_engine = (IBusNepaliEngine *)engine;
    log_message("[C LOG] process_key_event: keyval=%u", keyval);

    if (modifiers & IBUS_RELEASE_MASK) {
        return FALSE;
    }

    if (modifiers & (IBUS_CONTROL_MASK | IBUS_MOD1_MASK)) return FALSE;

    gboolean has_preedit = (nepali_engine->preedit_string->len > 0);
    gboolean has_candidates = ibus_lookup_table_get_number_of_candidates(nepali_engine->table) > 0;

    if (has_candidates) {
        switch (keyval) {
            case IBUS_KEY_Up:   
                ibus_lookup_table_cursor_up(nepali_engine->table);   
                ibus_engine_update_lookup_table(engine, nepali_engine->table, TRUE); 
                return TRUE;
            case IBUS_KEY_Down: 
                ibus_lookup_table_cursor_down(nepali_engine->table); 
                ibus_engine_update_lookup_table(engine, nepali_engine->table, TRUE); 
                return TRUE;
        }
    }
    
    switch (keyval) {
        case IBUS_KEY_space:
        case IBUS_KEY_Return:
            if (has_preedit) {
                if (has_candidates) {
                    commit_candidate_by_index(nepali_engine, ibus_lookup_table_get_cursor_pos(nepali_engine->table));
                } else {
                    commit_selection(nepali_engine, nepali_engine->preedit_string->str);
                }
                return (keyval == IBUS_KEY_space);
            }
            return FALSE;

        case IBUS_KEY_Escape:
            if (has_preedit) {
                clear_preedit(nepali_engine);
                return TRUE;
            }
            return FALSE;

        case IBUS_KEY_BackSpace:
            if (has_preedit) {
                g_string_truncate(nepali_engine->preedit_string, nepali_engine->preedit_string->len - 1);
                update_preedit_and_lookup(nepali_engine);
                return TRUE;
            }
            return FALSE;
        
        default:
            if (has_candidates && keyval >= '1' && keyval <= '9') {
                commit_candidate_by_index(nepali_engine, keyval - '1');
                return TRUE;
            }

            if (keyval >= 32 && keyval <= 126) {
                g_string_append_c(nepali_engine->preedit_string, (gchar)keyval);
                update_preedit_and_lookup(nepali_engine);
                return TRUE;
            }
    }

    return FALSE;
}

static void ibus_nepali_engine_candidate_clicked(IBusEngine *engine, guint index, guint button, guint state) {
    commit_candidate_by_index((IBusNepaliEngine *)engine, index);
}

// --- Main Function ---
int main(int argc, char **argv) {
    log_message("[C LOG] ========== Engine Starting ==========");
    log_message("[C LOG] argc=%d", argc);
    for (int i = 0; i < argc; i++) {
        log_message("[C LOG] argv[%d]=%s", i, argv[i]);
    }

    // Initialize IBus
    ibus_init();
    
    // Connect to IBus bus
    IBusBus *bus = ibus_bus_new();
    if (!ibus_bus_is_connected(bus)) {
        log_message("[C LOG] ERROR: Cannot connect to IBus bus!");
        return 1;
    }
    log_message("[C LOG] Connected to IBus bus successfully");
    
    // Create factory
    IBusFactory *factory = ibus_factory_new(ibus_bus_get_connection(bus));
    ibus_factory_add_engine(factory, "nepali-smart", IBUS_TYPE_NEPALI_ENGINE);
    log_message("[C LOG] Factory created and engine added");

    // CRITICAL FIX: Only register component if called WITH --ibus
    // When called without --ibus, just set up the factory and start the main loop
    if (argc > 1 && strcmp(argv[1], "--ibus") == 0) {
        log_message("[C LOG] Running in --ibus mode, registering component");
        
        // Request the service name so IBus daemon can find us
        guint result = ibus_bus_request_name(bus, "org.freedesktop.IBus.NepaliSmart", 0);
        log_message("[C LOG] Request name result: %u", result);
        
        if (result == 0) {
            log_message("[C LOG] ERROR: Failed to request name!");
            return 1;
        }
    } else {
        log_message("[C LOG] Running in standalone mode (no --ibus flag)");
        
        // When not started by IBus, we need to register our component
        IBusComponent *component = ibus_component_new(
            "org.freedesktop.IBus.NepaliSmart",
            "Nepali Smart IME",
            "1.0",
            "MIT",
            "Sabin",
            "https://github.com/yourusername/nepali-smart-ime",
            "/usr/lib/ibus/engines/nepali-smart --ibus",
            "ibus-keyboard"
        );

        IBusEngineDesc *desc = ibus_engine_desc_new(
            "nepali-smart",
            "Nepali (Smart)",
            "An intelligent, learning Nepali IME",
            "ne",
            "MIT",
            "Sabin",
            "/usr/share/icons/hicolor/scalable/apps/ibus-keyboard.svg",
            "us"
        );

        ibus_component_add_engine(component, desc);
        
        gboolean reg_result = ibus_bus_register_component(bus, component);
        log_message("[C LOG] Component registration result: %d", reg_result);
        
        if (!reg_result) {
            log_message("[C LOG] ERROR: Failed to register component!");
        }
        
        g_object_unref(component);
    }

    log_message("[C LOG] Starting main loop...");
    ibus_main();
    
    log_message("[C LOG] Engine shutting down");
    return 0;
}