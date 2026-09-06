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

# Relative-entropy pruning threshold for the browser model.  Higher = smaller
# and less accurate; see docs/WASM.md for the measured curve.  3e-2 keeps 47%
# of trigram transitions: 4.92 MB Brotli at 81.07% Aksharantar top-1.
TRIGRAM_THRESHOLD ?= 3e-2
.PHONY: all release debug install uninstall reinstall clean reset-learning restart-ibus help wasm wasm-clean wasm-serve release-upload train pack
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
	@sudo install -m 755 $(TARGET_DIR)/$(C_ENGINE_NAME) $(IBUS_ENGINE_DIR)/$(C_ENGINE_NAME).new
	@sudo mv -f $(IBUS_ENGINE_DIR)/$(C_ENGINE_NAME).new $(IBUS_ENGINE_DIR)/$(C_ENGINE_NAME)
	@sudo install -m 755 $(TARGET_DIR)/$(RUST_LIB_NAME) $(LIB_DIR)/$(RUST_LIB_NAME).new
	@sudo mv -f $(LIB_DIR)/$(RUST_LIB_NAME).new $(LIB_DIR)/$(RUST_LIB_NAME)
	@echo "  > Installing IBus component file..."
	@sudo cp devanagari-smart.xml $(IBUS_COMPONENT_DIR)/
	@echo "  > Installing model artifacts..."
	@if [ -f data/akshar.model ]; then \
		echo "    * Installing unified model (data/akshar.model)..."; \
		sudo cp data/akshar.model $(DATA_DIR)/akshar.model; \
		echo "    * Purging legacy model files from $(DATA_DIR)..."; \
		sudo rm -f $(DATA_DIR)/translit_model.bin $(DATA_DIR)/word_freq_text.bin $(DATA_DIR)/reranker_weights_sparse.bin $(DATA_DIR)/reranker_weights.json $(DATA_DIR)/crf_model.bin $(DATA_DIR)/roman_lexicon.bin; \
	else \
		echo "    * Installing legacy model artifacts..."; \
		for f in data/translit_model.bin data/word_freq_text.bin data/reranker_weights_sparse.bin data/roman_lexicon.bin; do \
			if [ -f "$$f" ]; then sudo cp "$$f" $(DATA_DIR)/; fi; \
		done; \
	fi
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

train:  ## Run the end-to-end one-shot model training pipeline (full 3.59M, chunked, ~40min).
	@cargo run --release --bin train -- --reranker-pairs 0 --epochs 5 --iterations 12

train-quick:  ## Fast training (100k pairs, ~10min).
	@cargo run --release --bin train -- --reranker-pairs 100000 --epochs 3 --iterations 12

pack:  ## Pack model binaries into unified data/akshar.model container.
	@cargo run --release --bin pack_model

web-model:  ## Build the compact browser model (data/akshar_wasm.model, ~4.9 MB Brotli).
	@echo "Pruning the syllable trigram LM (half the model's bytes)..."
	@cargo run --release --bin prune_lm -- \
		--model data/akshar.model \
		--trigram-threshold $(TRIGRAM_THRESHOLD) \
		--out data/akshar_wasm.model
	@if command -v brotli >/dev/null 2>&1; then \
		brotli -q 11 -c data/akshar_wasm.model | wc -c \
			| awk '{printf "Brotli wire size: %.2f MB\n", $$1/1048576}'; \
	fi

clean:  ## Remove all build artifacts and temporary files.
	@echo "Cleaning build artifacts..."
	@cargo clean
	@rm -f data/*.tmp data/smoke.model data/akshar_pruned.model data/akshar_quantized.model

reset-learning:  ## Delete the user's learned dictionary (start fresh).
	@echo "Removing user learning data..."
	@rm -f $${XDG_CONFIG_HOME:-$$HOME/.config}/akshar-devanagari/user_dictionary.bin
	@echo "Done."

wasm:  ## Build WASM package (wasm/pkg + JS wrapper).
	@echo "Building WASM package..."
	@bash wasm/build.sh

wasm-clean:  ## Remove WASM build artifacts.
	@echo "Cleaning WASM artifacts..."
	@rm -rf wasm/pkg

wasm-serve: wasm  ## Build WASM and serve demo at http://localhost:PORT/web/ (default 8000)
	@PORT=$${PORT:-8000}; \
	ORIG=$$PORT; \
	for p in $$PORT 8001 8002 8003 8004 8005 8006 8007 8008 8009 8010; do \
	  if ! ss -tln 2>/dev/null | grep -q ":$$p " && ! ss -tln6 2>/dev/null | grep -q ":$$p "; then PORT=$$p; break; fi; \
	done; \
	if [ "$$PORT" != "$$ORIG" ]; then echo "Port $$ORIG in use, using $$PORT instead"; fi; \
	echo "Serving demo at http://localhost:$$PORT/web/ (Ctrl+C to stop)"; \
	echo "  (override with: make wasm-serve PORT=9000)"; \
	python3 -m http.server $$PORT

release-upload:  ## Upload locally built model artifacts to a GitHub release (TAG=vX.Y.Z required).
	@if [ -z "$(TAG)" ]; then echo "usage: make release-upload TAG=vX.Y.Z"; exit 1; fi
	@if [ -f data/akshar.model ]; then \
		echo "Uploading unified model data/akshar.model..."; \
		gh release view $(TAG) >/dev/null 2>&1 || gh release create $(TAG) --generate-notes --verify-tag; \
		gh release upload $(TAG) data/akshar.model --clobber; \
	elif [ -f data/translit_model.bin ] && [ -f data/word_freq_text.bin ]; then \
		echo "Uploading legacy model binaries..."; \
		gh release view $(TAG) >/dev/null 2>&1 || gh release create $(TAG) --generate-notes --verify-tag; \
		gh release upload $(TAG) data/translit_model.bin data/word_freq_text.bin --clobber; \
	else \
		echo "No model artifacts found in data/. Run 'make train' or 'make pack' first."; exit 1; \
	fi
	@echo "Uploaded model artifacts to release $(TAG)."

# --- Help ---

help:  ## Show this help.
	@echo "Akshar Devanagari IME Makefile"
	@echo "-------------------------"
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'