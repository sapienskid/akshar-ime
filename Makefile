
# File: Makefile
# Makefile for Production-Grade IBus Engine Deployment
RUST_LIB_NAME = libnepali_smart_ime.so
C_ENGINE_NAME = nepali-smart
TARGET_DIR = target/release

# Standard C compiler and linker flags for IBus and Jansson
CFLAGS = $(shell pkg-config --cflags ibus-1.0 jansson) -fPIC -g
LDFLAGS = $(shell pkg-config --libs ibus-1.0 jansson)

# Standard System Paths
STANDARD_LIB_PATH = /usr/lib
IBUS_ENGINE_PATH = /usr/lib/ibus/engines
IBUS_COMPONENT_PATH = /usr/share/ibus/component

# User data paths for dictionary management (respect XDG where possible)
XDG_CONFIG ?= $(HOME)/.config
XDG_DATA ?= $(HOME)/.local/share
DICTIONARY_REL_PATH = nepali-smart-ime/user_dictionary.bin
DICT_LOCATIONS = $(XDG_CONFIG)/$(DICTIONARY_REL_PATH) $(XDG_DATA)/$(DICTIONARY_REL_PATH) $(HOME)/$(DICTIONARY_REL_PATH)
BACKUP_DIR ?= $(XDG_DATA)/nepali-smart-ime/backups


.PHONY: all release debug test install clean uninstall check-install

all: release

release: rust_lib c_engine

debug:
	cargo build

rust_lib:
	cargo build --release

c_engine: rust_lib
	gcc $(CFLAGS) -o $(TARGET_DIR)/$(C_ENGINE_NAME) src/ibus_engine.c \
		-L$(TARGET_DIR) -lnepali_smart_ime $(LDFLAGS) -Wl,-rpath,/usr/lib

# Installation with proper steps
install: release
	@echo "======================================================================"
	@echo "Installing Nepali Smart IME..."
	@echo "======================================================================"
	@echo ""
	@echo "Step 1: Stopping IBus daemon..."
	-ibus exit 2>/dev/null || true
	@sleep 1
	
	@echo ""
	@echo "Step 2: Creating directories..."
	sudo mkdir -p $(IBUS_ENGINE_PATH)
	sudo mkdir -p $(IBUS_COMPONENT_PATH)
	
	@echo ""
	@echo "Step 3: Installing engine binary..."
	sudo cp $(TARGET_DIR)/$(C_ENGINE_NAME) $(IBUS_ENGINE_PATH)/
	sudo chmod +x $(IBUS_ENGINE_PATH)/$(C_ENGINE_NAME)
	
	@echo ""
	@echo "Step 4: Installing Rust library..."
	sudo cp $(TARGET_DIR)/$(RUST_LIB_NAME) $(STANDARD_LIB_PATH)/
	
	@echo ""
	@echo "Step 5: Installing component XML..."
	sudo cp nepali-smart.xml $(IBUS_COMPONENT_PATH)/
	
	@echo ""
	@echo "Step 6: Updating linker cache..."
	sudo ldconfig
	
	@echo ""
	@echo "Step 7: Clearing IBus cache..."
		rm -f ~/.cache/ibus/bus/* 2>/dev/null || true
	
	@echo ""
	@echo "Step 8: Restarting IBus daemon..."
	@sleep 1
	ibus-daemon --daemonize --replace --xim
	@sleep 2
	
	@echo ""
	@echo "======================================================================"
	@echo "Installation complete!"
	@echo "======================================================================"
	@echo ""
	@echo "Next steps:"
	@echo "1. Run 'make check-install' to verify the installation"
	@echo "2. Go to your system settings > Keyboard > Input Sources"
	@echo "3. Add 'Nepali (Smart)' to your input methods"
	@echo "4. If it doesn't appear, try:"
	@echo "   - Logging out and logging back in"
	@echo "   - Or run: ibus restart"
	@echo ""

# Check installation
check-install:
	@echo "======================================================================"
	@echo "Checking Nepali Smart IME Installation..."
	@echo "======================================================================"
	@echo ""
	@echo "1. Checking engine binary:"
	@if [ -x "$(IBUS_ENGINE_PATH)/$(C_ENGINE_NAME)" ]; then \
		echo "   ✓ Engine binary exists and is executable"; \
		ls -lh $(IBUS_ENGINE_PATH)/$(C_ENGINE_NAME); \
	else \
		echo "   ✗ Engine binary not found or not executable!"; \
	fi
	@echo ""
	@echo "2. Checking Rust library:"
	@if [ -f "$(STANDARD_LIB_PATH)/$(RUST_LIB_NAME)" ]; then \
		echo "   ✓ Rust library exists"; \
		ls -lh $(STANDARD_LIB_PATH)/$(RUST_LIB_NAME); \
	else \
		echo "   ✗ Rust library not found!"; \
	fi
	@echo ""
	@echo "3. Checking component XML:"
	@if [ -f "$(IBUS_COMPONENT_PATH)/nepali-smart.xml" ]; then \
		echo "   ✓ Component XML exists"; \
		ls -lh $(IBUS_COMPONENT_PATH)/nepali-smart.xml; \
	else \
		echo "   ✗ Component XML not found!"; \
	fi
	@echo ""
	@echo "4. Checking if IBus daemon is running:"
	@if pgrep -x "ibus-daemon" > /dev/null; then \
		echo "   ✓ IBus daemon is running"; \
		pgrep -ax ibus-daemon; \
	else \
		echo "   ✗ IBus daemon is not running!"; \
		echo "   Run: ibus-daemon --daemonize --replace --xim"; \
	fi
	@echo ""
	@echo "5. Listing available IBus engines:"
	@ibus list-engine | grep -i nepali || echo "   ⚠ Nepali engine not found in IBus engine list"
	@echo ""
	@echo "6. Testing engine executable:"
	@echo "   Running: $(IBUS_ENGINE_PATH)/$(C_ENGINE_NAME) --help"
	@-$(IBUS_ENGINE_PATH)/$(C_ENGINE_NAME) --help 2>&1 | head -5 || \
		echo "   (This is expected if --help is not implemented)"
	@echo ""
	@echo "7. Checking log file:"
	@if [ -f "/run/media/sapiens/Development/Development/nepali-ime/target/c_engine.log" ]; then \
		echo "   Log file exists. Last 10 lines:"; \
		tail -10 /run/media/sapiens/Development/Development/nepali-ime/target/c_engine.log; \
	else \
		echo "   ⚠ Log file not found yet (will be created when engine starts)"; \
	fi
	@echo ""

# Test the engine manually
test-engine:
	@echo "Testing engine manually (press Ctrl+C to stop)..."
	$(IBUS_ENGINE_PATH)/$(C_ENGINE_NAME) --ibus

# Uninstall
uninstall:
	@echo "======================================================================"
	@echo "Uninstalling Nepali Smart IME..."
	@echo "======================================================================"
	@echo ""
	@echo "Stopping IBus daemon..."
	-ibus exit 2>/dev/null || true
	@sleep 1
	
	@echo "Removing installed files..."
	sudo rm -f $(IBUS_ENGINE_PATH)/$(C_ENGINE_NAME)
	sudo rm -f $(STANDARD_LIB_PATH)/$(RUST_LIB_NAME)
	sudo rm -f $(IBUS_COMPONENT_PATH)/nepali-smart.xml
	
	@echo "Updating linker cache..."
	sudo ldconfig
	
	@echo "Clearing IBus cache..."
	rm -f ~/.cache/ibus/bus/* 2>/dev/null || true
	
	@echo "Restarting IBus daemon..."
	ibus-daemon --daemonize --replace --xim
	
	@echo ""
	@echo "Uninstallation complete!"
	@echo ""

clean:
	cargo clean
	rm -f $(TARGET_DIR)/$(C_ENGINE_NAME)

## ------------------ Dictionary & Cache Utilities ------------------ ##
.PHONY: reset-dictionary clear-ibus-cache factory-reset reinstall backup-dictionary

# Back up and remove any user dictionary found in typical locations
reset-dictionary:
	@echo "======================================================================"
	@echo "Backing up and removing user dictionary(s)"
	@echo "======================================================================"
	@mkdir -p $(BACKUP_DIR)
	@TIMESTAMP=`date +%Y%m%d-%H%M%S` && \
	for f in $(DICT_LOCATIONS); do \
		if [ -f "$$f" ]; then \
			echo "Backing up $$f to $(BACKUP_DIR)/user_dictionary.bin.$$TIMESTAMP"; \
			cp "$$f" "$(BACKUP_DIR)/user_dictionary.bin.$$TIMESTAMP"; \
			rm -f "$$f"; \
		else \
			echo "No dictionary found at $$f"; \
		fi; \
	done

# Explicit cache clearing for ibus
clear-ibus-cache:
	@echo "Clearing IBus cache files..."
	@rm -f ~/.cache/ibus/bus/* 2>/dev/null || true
	@rm -rf ~/.cache/ibus/* 2>/dev/null || true
	@echo "Restarting IBus daemon..."
	-ibus exit 2>/dev/null || true
	@sleep 1
	@ibus-daemon --daemonize --replace --xim || true
	@sleep 1
	@echo "IBus cache cleared and daemon restarted."

# Convenience: perform both cache and dictionary reset
factory-reset: clear-ibus-cache reset-dictionary
	@echo "Factory reset complete. Please re-add the input method if necessary."

# Reinstall shortcut
reinstall: uninstall release install
	@echo "Reinstall completed"

# Backup dictionary only
backup-dictionary:
	@mkdir -p $(BACKUP_DIR)
	@TIMESTAMP=`date +%Y%m%d-%H%M%S` && \
	for f in $(DICT_LOCATIONS); do \
		if [ -f "$$f" ]; then \
			echo "Copying $$f to $(BACKUP_DIR)/user_dictionary.bin.$$TIMESTAMP"; \
			cp "$$f" "$(BACKUP_DIR)/user_dictionary.bin.$$TIMESTAMP"; \
		else \
			echo "No dictionary found at $$f"; \
		fi; \
	done
