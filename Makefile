# ==============================================================================
# Makefile for Nepali Smart IME
# ==============================================================================

# --- Variables ---
RUST_LIB_NAME := libnepali_smart_ime.so
C_ENGINE_NAME := nepali-smart
TARGET_DIR    := target/release

# Compiler and Linker Flags (discovered via pkg-config for portability)
CFLAGS   := $(shell pkg-config --cflags ibus-1.0 jansson) -fPIC -O2
LDFLAGS  := $(shell pkg-config --libs ibus-1.0 jansson)

# System Paths
PREFIX            ?= /usr
LIB_DIR           := $(PREFIX)/lib
IBUS_ENGINE_DIR   := $(PREFIX)/lib/ibus/engines
IBUS_COMPONENT_DIR:= $(PREFIX)/share/ibus/component

.PHONY: all release debug install uninstall reinstall clean help

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
		-L$(TARGET_DIR) -lnepali_smart_ime $(LDFLAGS) -Wl,-rpath,$(LIB_DIR)

# --- Installation & Management ---

install: release  ## Compile and install the engine to system directories.
	@echo "Installing Nepali Smart IME..."
	@echo "  > Stopping IBus daemon..."
	@-ibus exit 2>/dev/null || true
	@echo "  > Creating system directories..."
	@sudo mkdir -p $(IBUS_ENGINE_DIR)
	@sudo mkdir -p $(IBUS_COMPONENT_DIR)
	@echo "  > Installing engine binary and library..."
	@sudo cp $(TARGET_DIR)/$(C_ENGINE_NAME) $(IBUS_ENGINE_DIR)/
	@sudo cp $(TARGET_DIR)/$(RUST_LIB_NAME) $(LIB_DIR)/
	@echo "  > Installing IBus component file..."
	@sudo cp nepali-smart.xml $(IBUS_COMPONENT_DIR)/
	@echo "  > Updating linker cache..."
	@sudo ldconfig
	@echo "  > Clearing IBus cache and restarting daemon..."
	@rm -f ~/.cache/ibus/bus/* 2>/dev/null || true
	@ibus-daemon --daemonize --replace --xim
	@echo "\nInstallation complete. Please add 'Nepali (Smart)' in your system's keyboard settings."

uninstall:  ## Remove the engine from the system.
	@echo "Uninstalling Nepali Smart IME..."
	@echo "  > Stopping IBus daemon..."
	@-ibus exit 2>/dev/null || true
	@echo "  > Removing system files..."
	@sudo rm -f $(IBUS_ENGINE_DIR)/$(C_ENGINE_NAME)
	@sudo rm -f $(LIB_DIR)/$(RUST_LIB_NAME)
	@sudo rm -f $(IBUS_COMPONENT_DIR)/nepali-smart.xml
	@echo "  > Updating linker cache..."
	@sudo ldconfig
	@echo "  > Restarting IBus daemon..."
	@ibus-daemon --daemonize --replace --xim
	@echo "\nUninstallation complete."

reinstall: uninstall install  ## Run uninstall and then install.

clean:  ## Remove all build artifacts.
	@echo "Cleaning build artifacts..."
	@cargo clean

# --- Help ---

help:  ## Show this help message.
	@echo "Nepali Smart IME Makefile"
	@echo "-------------------------"
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'