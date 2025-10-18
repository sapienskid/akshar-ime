.PHONY: all release debug test install profile clean

# Default target
all: release

# Optimized production binary (< 1.5 MB)
release:
	@echo "Building optimized release binary..."
	@cargo build --release
	@echo "Binary size:"
	@size target/release/ime_engine

# Development build with debug symbols
debug:
	@cargo build

# Run unit tests and benchmarks
test:
	@cargo test -- --nocapture

# System-wide deployment (example)
install: release
	@echo "Installing ime_engine to /usr/local/bin..."
	@sudo cp target/release/ime_engine /usr/local/bin/
	@echo "Installing m17n definition..."
	@sudo mkdir -p /usr/share/m17n/
	@sudo cp m17n/ne-smart.mim /usr/share/m17n/ne-smart.mim
	@echo "Installation complete. Restart your input method framework."

# Build for performance profiling
profile:
	@cargo build --release --features="flame_it"

# Clean build artifacts
clean:
	@cargo clean