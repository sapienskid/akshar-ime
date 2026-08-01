# ==============================================================================
# Makefile for Akshar Devanagari IME
# ==============================================================================

# --- Variables ---
RUST_LIB_NAME := libakshar_ime.so
C_ENGINE_NAME := devanagari-smart
TARGET_DIR    := target/release

# Compiler and Linker Flags (discovered via pkg-config for portability)
CFLAGS   := $(shell pkg-config --cflags ibus-1.0 jansson) -fPIC -O2
LDFLAGS  := $(shell pkg-config --libs ibus-1.0 jansson)

# System Paths
PREFIX            ?= /usr
LIB_DIR           := $(PREFIX)/lib
IBUS_ENGINE_DIR   := $(PREFIX)/lib/ibus/engines
IBUS_COMPONENT_DIR:= $(PREFIX)/share/ibus/component
DATA_DIR          := $(PREFIX)/share/akshar-ime

.PHONY: all release debug install uninstall reinstall clean reset-learning restart-ibus help

# --- Main Targets ---

all: release  ## Build the engine for release (default).

release: rust_lib c_engine  ## Build the Rust library and C engine in release mode.

debug:  ## Build the Rust library in debug mode.
	@echo "Building Rust library in debug mode..."
	@cargo build

test:  ## Run the Rust test suite.
	@echo "Running Rust tests..."
	@cargo test

# --- Build Steps ---

rust_lib:
	@echo "Building Rust library in release mode..."
	@cargo build --release

c_engine: rust_lib
	@echo "Building C engine against release library..."
	@$(CC) $(CFLAGS) -o $(TARGET_DIR)/$(C_ENGINE_NAME) src/ibus_engine.c \
		-L$(TARGET_DIR) -lakshar_ime $(LDFLAGS) -Wl,-rpath,$(LIB_DIR)


install:  ## Compile (if needed) and install the engine to system directories.
	@if [ ! -f $(TARGET_DIR)/libakshar_ime.so ] || [ ! -f $(TARGET_DIR)/$(C_ENGINE_NAME) ]; then \
		echo "  > Building release artifacts first..."; \
		$(MAKE) release; \
	fi
	@echo "Installing Akshar Devanagari IME..."
	@echo "  > Creating system directories..."
	@sudo mkdir -p $(IBUS_ENGINE_DIR)
	@sudo mkdir -p $(IBUS_COMPONENT_DIR)
	@sudo mkdir -p $(DATA_DIR)
	@echo "  > Installing engine binary and library..."
	@sudo cp $(TARGET_DIR)/$(C_ENGINE_NAME) $(IBUS_ENGINE_DIR)/
	@sudo cp $(TARGET_DIR)/$(RUST_LIB_NAME) $(LIB_DIR)/
	@echo "  > Installing IBus component file..."
	@sudo cp devanagari-smart.xml $(IBUS_COMPONENT_DIR)/
	@echo "  > Installing model artifacts..."
	@sudo cp data/translit_model.bin data/roman_lexicon.bin data/reranker_weights.json $(DATA_DIR)/
	@echo "  > Updating linker cache..."
	@sudo ldconfig
	@echo "\nInstallation complete. Run 'make restart-ibus' (no sudo) to reload IBus,"
	@echo "then add 'Devanagari (Akshar)' in Settings > Keyboard > Input Sources."

restart-ibus:  ## Restart the user's IBus daemon (run WITHOUT sudo).
	@echo "Restarting IBus..."
	@-timeout 5 ibus restart 2>/dev/null || true
	@rm -f ~/.cache/ibus/bus/* 2>/dev/null || true
	@echo "Done. If the input source still doesn't appear, log out and back in."


uninstall:  ## Remove the engine from the system.
	@echo "Uninstalling Akshar Devanagari IME..."
	@echo "  > Removing system files..."
	@sudo rm -f $(IBUS_ENGINE_DIR)/$(C_ENGINE_NAME)
	@sudo rm -f $(LIB_DIR)/$(RUST_LIB_NAME)
	@sudo rm -f $(IBUS_COMPONENT_DIR)/devanagari-smart.xml
	@sudo rm -rf $(DATA_DIR)
	@echo "  > Updating linker cache..."
	@sudo ldconfig
	@echo "\nUninstallation complete. Run 'make restart-ibus' (no sudo) to reload IBus."

reinstall: uninstall install  ## Run uninstall and then install.

clean:  ## Remove all build artifacts.
	@echo "Cleaning build artifacts..."
	@cargo clean

reset-learning:  ## Delete the user's learned dictionary (start fresh).
	@echo "Removing user learning data..."
	@rm -f $${XDG_CONFIG_HOME:-$$HOME/.config}/akshar-devanagari/user_dictionary.bin
	@echo "Done."

# --- Help ---

help:  ## Show this help.
	@echo "Akshar Devanagari IME Makefile"
	@echo "-------------------------"
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'