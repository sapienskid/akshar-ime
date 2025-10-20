#include <ibus.h>
#include <jansson.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>

// --- Rust FFI function declarations ---
void nepali_ime_engine_init(void);
void nepali_ime_engine_destroy(void);
char *nepali_ime_get_suggestions(const char *prefix);
void nepali_ime_confirm_word(const char *roman, const char *nepali);
void nepali_ime_free_string(char *s);

// --- GObject Boilerplate ---
typedef struct _IBusNepaliEngine IBusNepaliEngine;
typedef struct _IBusNepaliEngineClass IBusNepaliEngineClass;
struct _IBusNepaliEngine
{
    IBusEngine parent;
    IBusLookupTable *table;
    GString *preedit_string;
};
struct _IBusNepaliEngineClass
{
    IBusEngineClass parent;
};
static guint g_engine_instance_count = 0;
static void ibus_nepali_engine_class_init(IBusNepaliEngineClass *klass);
static void ibus_nepali_engine_init_instance(IBusNepaliEngine *engine);
static void ibus_nepali_engine_finalize(GObject *object);
static gboolean ibus_nepali_engine_process_key_event(IBusEngine *engine, guint keyval, guint keycode, guint modifiers);
static void ibus_nepali_engine_candidate_clicked(IBusEngine *engine, guint index, guint button, guint state);
G_DEFINE_TYPE(IBusNepaliEngine, ibus_nepali_engine, IBUS_TYPE_ENGINE)
#define IBUS_TYPE_NEPALI_ENGINE (ibus_nepali_engine_get_type())

// --- Initialization and Finalization ---
static void ibus_nepali_engine_class_init(IBusNepaliEngineClass *klass)
{
    IBusEngineClass *engine_class = IBUS_ENGINE_CLASS(klass);
    GObjectClass *object_class = G_OBJECT_CLASS(klass);
    engine_class->process_key_event = ibus_nepali_engine_process_key_event;
    engine_class->candidate_clicked = ibus_nepali_engine_candidate_clicked;
    object_class->finalize = ibus_nepali_engine_finalize;
}
static void ibus_nepali_engine_init_instance(IBusNepaliEngine *engine)
{
    engine->preedit_string = g_string_new("");
    engine->table = ibus_lookup_table_new(10, 0, TRUE, TRUE);
    g_object_ref_sink(engine->table);
    if (g_engine_instance_count == 0)
    {
        nepali_ime_engine_init();
    }
    g_engine_instance_count++;
}
static void ibus_nepali_engine_init(IBusNepaliEngine *engine) { ibus_nepali_engine_init_instance(engine); }
static void ibus_nepali_engine_finalize(GObject *object)
{
    g_engine_instance_count--;
    if (g_engine_instance_count == 0)
    {
        nepali_ime_engine_destroy();
    }
    G_OBJECT_CLASS(ibus_nepali_engine_parent_class)->finalize(object);
}

// --- Core IME Logic ---
static void clear_preedit(IBusNepaliEngine *nepali_engine)
{
    g_string_set_size(nepali_engine->preedit_string, 0);
    ibus_engine_hide_preedit_text((IBusEngine *)nepali_engine);
    ibus_engine_hide_lookup_table((IBusEngine *)nepali_engine);
}

static void update_preedit_and_lookup(IBusNepaliEngine *nepali_engine)
{
    IBusEngine *engine = (IBusEngine *)nepali_engine;
    const char *preedit_str = nepali_engine->preedit_string->str;

    if (strlen(preedit_str) == 0)
    {
        clear_preedit(nepali_engine);
        return;
    }

    IBusText *preedit_text = ibus_text_new_from_string(preedit_str);
    ibus_engine_update_preedit_text(engine, preedit_text, strlen(preedit_str), TRUE);
    ibus_lookup_table_clear(nepali_engine->table);

    char *suggestions_json = nepali_ime_get_suggestions(preedit_str);
    json_error_t error;
    json_t *root = json_loads(suggestions_json, 0, &error);

    if (root && json_is_array(root))
    {
        size_t i;
        json_t *value;
        json_array_foreach(root, i, value)
        {
            if (json_is_string(value))
            {
                IBusText *candidate_text = ibus_text_new_from_string(json_string_value(value));
                ibus_lookup_table_append_candidate(nepali_engine->table, candidate_text);
            }
        }
        json_decref(root);
    }
    nepali_ime_free_string(suggestions_json);

    if (ibus_lookup_table_get_number_of_candidates(nepali_engine->table) > 0)
    {
        ibus_engine_update_lookup_table(engine, nepali_engine->table, TRUE);
    }
    else
    {
        ibus_engine_hide_lookup_table(engine);
    }
}

// Commits the currently selected candidate or the top suggestion if none is selected.
static void commit_best_candidate(IBusNepaliEngine *nepali_engine)
{
    if (nepali_engine->preedit_string->len == 0)
        return;

    const char *preedit_for_confirm = g_strdup(nepali_engine->preedit_string->str);
    IBusText *commit_text = NULL;

    // First, try to get the user-selected candidate
    guint index = ibus_lookup_table_get_cursor_pos(nepali_engine->table);
    commit_text = ibus_lookup_table_get_candidate(nepali_engine->table, index);
    if (commit_text)
    {
        g_object_ref(commit_text); // Increment ref count because we are using it
    }

    // If no candidate is selected, fetch the top suggestion directly from Rust
    if (!commit_text)
    {
        char *suggestions_json = nepali_ime_get_suggestions(preedit_for_confirm);
        json_error_t error;
        json_t *root = json_loads(suggestions_json, 0, &error);
        if (root && json_is_array(root) && json_array_size(root) > 0)
        {
            json_t *first = json_array_get(root, 0);
            if (json_is_string(first))
            {
                commit_text = ibus_text_new_from_string(json_string_value(first));
            }
        }
        if (root)
            json_decref(root);
        nepali_ime_free_string(suggestions_json);
    }

    if (commit_text && commit_text->text)
    {
        ibus_engine_commit_text((IBusEngine *)nepali_engine, commit_text);
        nepali_ime_confirm_word(preedit_for_confirm, commit_text->text);
    }

    if (commit_text)
    {
        g_object_unref(commit_text);
    }
    g_free((gpointer)preedit_for_confirm);
    clear_preedit(nepali_engine);
}

static void ibus_nepali_engine_candidate_clicked(IBusEngine *engine, guint index, guint button, guint state)
{
    ibus_lookup_table_set_cursor_pos(((IBusNepaliEngine *)engine)->table, index);
    commit_best_candidate((IBusNepaliEngine *)engine);
}

// --- RE-ARCHITECTED: The main key event processor ---
static gboolean ibus_nepali_engine_process_key_event(IBusEngine *engine, guint keyval, guint keycode, guint modifiers)
{
    IBusNepaliEngine *nepali_engine = (IBusNepaliEngine *)engine;

    if ((modifiers & IBUS_RELEASE_MASK) || (modifiers & (IBUS_CONTROL_MASK | IBUS_MOD1_MASK)))
    {
        return FALSE;
    }

    gboolean has_preedit = (nepali_engine->preedit_string->len > 0);
    gboolean has_candidates = ibus_lookup_table_get_number_of_candidates(nepali_engine->table) > 0;

    // --- Punctuation and Symbol Handling ---
    // Check for symbols that should immediately commit.
    if (keyval == IBUS_KEY_period || keyval == IBUS_KEY_question || keyval == IBUS_KEY_comma ||
        (keyval >= IBUS_KEY_0 && keyval <= IBUS_KEY_9))
    {
        if (has_preedit)
        {
            commit_best_candidate(nepali_engine);
        }
        // Now, transliterate and commit the symbol itself
        char symbol_str[2] = {(char)keyval, '\0'};
        char *suggestions_json = nepali_ime_get_suggestions(symbol_str);
        json_error_t error;
        json_t *root = json_loads(suggestions_json, 0, &error);
        if (root && json_is_array(root) && json_array_size(root) > 0)
        {
            json_t *first = json_array_get(root, 0);
            if (json_is_string(first))
            {
                IBusText *text = ibus_text_new_from_string(json_string_value(first));
                ibus_engine_commit_text(engine, text);
                g_object_unref(text);
            }
        }
        if (root)
            json_decref(root);
        nepali_ime_free_string(suggestions_json);
        return TRUE; // Consume the key event
    }

    // --- Candidate Navigation ---
    if (has_candidates)
    {
        switch (keyval)
        {
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

    // --- Keypress Processing ---
    switch (keyval)
    {
    case IBUS_KEY_Return:
    case IBUS_KEY_space:
    case IBUS_KEY_Tab:
        if (has_preedit)
        {
            commit_best_candidate(nepali_engine);
            return TRUE; // Consume the event to prevent extra space/enter.
        }
        return FALSE; // No preedit, so pass the key to the application.

    case IBUS_KEY_Escape:
        if (has_preedit)
        {
            clear_preedit(nepali_engine);
            return TRUE;
        }
        return FALSE;

    case IBUS_KEY_BackSpace:
        if (has_preedit)
        {
            g_string_truncate(nepali_engine->preedit_string, nepali_engine->preedit_string->len - 1);
            update_preedit_and_lookup(nepali_engine);
            return TRUE;
        }
        return FALSE;
    }

    // --- Alphanumeric Input ---
    if ((keyval >= IBUS_KEY_a && keyval <= IBUS_KEY_z) || (keyval >= IBUS_KEY_A && keyval <= IBUS_KEY_Z) || (keyval >= IBUS_KEY_0 && keyval <= IBUS_KEY_9))
    {
        g_string_append_c(nepali_engine->preedit_string, (gchar)keyval);
        update_preedit_and_lookup(nepali_engine);
        return TRUE;
    }

    return FALSE;
}

// --- Main Function (unchanged) ---
int main(int argc, char **argv)
{
    ibus_init();
    IBusBus *bus = ibus_bus_new();
    if (!ibus_bus_is_connected(bus))
    {
        return 1;
    }
    IBusFactory *factory = ibus_factory_new(ibus_bus_get_connection(bus));
    ibus_factory_add_engine(factory, "nepali-smart", IBUS_TYPE_NEPALI_ENGINE);
    if (argc > 1 && strcmp(argv[1], "--ibus") == 0)
    {
        ibus_bus_request_name(bus, "org.freedesktop.IBus.NepaliSmart", 0);
    }
    ibus_main();
    return 0;
}