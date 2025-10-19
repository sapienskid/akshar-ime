.PHONY: all release debug test install uninstall profile clean

# Default target
all: release

# Optimized production binary
release:
	@echo "Building optimized release binary..."
	@cargo build --release

# Development build
debug:
	@cargo build

# Run tests
test:
	@cargo test -- --nocapture

# --- FINAL INSTALL TARGET ---
install: release
	@echo "Installing ime_engine binary to /usr/local/bin..."
	@sudo cp target/release/ime_engine /usr/local/bin/
	
	@echo "Installing Python loader script to /usr/local/bin..."
	@sudo cp ibus/nepali-smart-ime-loader.py /usr/local/bin/
	@echo "Making loader script executable..."
	@sudo chmod +x /usr/local/bin/nepali-smart-ime-loader.py
	
	@echo "Installing IBus component definition..."
	@sudo cp ibus/nepali_smart_ime.xml /usr/share/ibus/component/
	
	@echo "Installation complete. Please restart IBus."

# --- FINAL UNINSTALL TARGET ---
uninstall:
	@echo "Uninstalling ime_engine binary..."
	@sudo rm -f /usr/local/bin/ime_engine
	@echo "Uninstalling Python loader script..."
	@sudo rm -f /usr/local/bin/nepali-smart-ime-loader.py
	@echo "Uninstalling IBus component definition..."
	@sudo rm -f /usr/share/ibus/component/nepali_smart_ime.xml
	@echo "Uninstallation complete. Please restart IBus."

# Build for profiling
profile:
	@cargo build --release --features="flame_it"

# Clean build artifacts
clean:
	@cargo clean