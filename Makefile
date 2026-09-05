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

.PHONY: all release debug install uninstall reinstall clean reset-learning restart-ibus help wasm wasm-clean wasm-serve data data-raw data-vocab data-model data-store

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
	@for f in data/translit_model.bin data/word_freq_text.bin data/reranker_weights.json data/roman_lexicon.bin; do \
		if [ -f "$$f" ]; then sudo cp "$$f" $(DATA_DIR)/; fi; \
	done
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

# --- Data system (see data/README.md) — everything builds into data/ ---

data: data-raw data-clean data-vocab data-model  ## Build the full data chain: downloads → cleaned corpus → vocab → model.

data-raw:  ## Download source corpora: Aksharantar splits + Nepali Wikipedia + CC100 → data/raw (transient).
	@mkdir -p data/aksharantar data/raw
	@python3 data/pipeline/fetch_corpus.py data/aksharantar
	@if [ ! -f data/raw/newiki.txt ]; then \
		echo "  > Downloading + extracting Nepali Wikipedia..."; \
		curl -sL -o /tmp/newiki.xml.bz2 https://dumps.wikimedia.org/newiki/latest/newiki-latest-pages-articles.xml.bz2; \
		python3 data/pipeline/extract_wiki.py /tmp/newiki.xml.bz2 data/raw/newiki.txt; \
		rm -f /tmp/newiki.xml.bz2; \
	else echo "  > data/raw/newiki.txt already present"; fi
	@if [ ! -f data/raw/cc100ne.txt ]; then \
		echo "  > Downloading + filtering CC100 Nepali..."; \
		curl -sL -o /tmp/cc100-ne.txt.xz https://data.statmt.org/cc-100/ne.txt.xz; \
		python3 data/pipeline/filter_cc100.py /tmp/cc100-ne.txt.xz data/raw/cc100ne.txt; \
		rm -f /tmp/cc100-ne.txt.xz; \
	else echo "  > data/raw/cc100ne.txt already present"; fi

data-clean:  ## Compile all sources into the single cleaned corpus (data/store/corpus_clean.txt), then delete the raw inputs.
	@mkdir -p data/store
	@python3 data/pipeline/build_corpus.py data/store/corpus_clean.txt \
		--db data/store/nepali_text.db \
		data/raw/newiki.txt data/raw/cc100ne.txt data/raw/news.txt
	@rm -rf data/raw
	@echo "  > raw sources deleted; corpus_clean.txt is now the only stored text"

data-vocab:  ## Count the cleaned corpus → data/word_freq_text.bin.
	@cargo run --release --bin build_wordfreq_text -- data/store/corpus_clean.txt

data-model:  ## Train the EM model (train+valid) → data/translit_model.bin.
	@cargo run --release --bin train_model -- \
		--train data/aksharantar/nep_train.json \
		--extra data/aksharantar/nep_valid.json \
		--out data/translit_model.bin

data-store:  ## Scraping pipeline: incremental crawl + count + export, then re-clean corpus + vocab.
	@python3 data/pipeline/pipeline.py crawl
	@python3 data/pipeline/pipeline.py count
	@python3 data/pipeline/pipeline.py export --merge-base data/word_freq_text.bin
	@$(MAKE) data-clean data-vocab

release-upload:  ## Upload locally built model artifacts to a GitHub release (TAG=vX.Y.Z required).
	@if [ -z "$(TAG)" ]; then echo "usage: make release-upload TAG=vX.Y.Z"; exit 1; fi
	@test -f data/translit_model.bin || { echo "data/translit_model.bin missing — run 'make data-model' first"; exit 1; }
	@test -f data/word_freq_text.bin || { echo "data/word_freq_text.bin missing — run 'make data-vocab' first"; exit 1; }
	@gh release view $(TAG) >/dev/null 2>&1 || gh release create $(TAG) --generate-notes --verify-tag
	@gh release upload $(TAG) data/translit_model.bin data/word_freq_text.bin --clobber
	@echo "Uploaded model artifacts to release $(TAG)."

# --- Help ---

help:  ## Show this help.
	@echo "Akshar Devanagari IME Makefile"
	@echo "-------------------------"
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'