.PHONY: help setup build dist clean clean-all

# Python interpreter
PYTHON := python3
VENV := venv
BIN := $(VENV)/bin
PIP_VENV := $(BIN)/pip

# Project settings
PROJECT_NAME := wallpaper_generator
MAIN_SCRIPT := wallpaper_generator.py

# Build settings
BUILD_DIR := build
DIST_DIR := dist
BINARY_NAME := wallpaper-generator
SPEC_FILE := $(PROJECT_NAME).spec

# Colors for output
COLOR_RESET := \033[0m
COLOR_BOLD := \033[1m
COLOR_GREEN := \033[32m
COLOR_YELLOW := \033[33m
COLOR_BLUE := \033[34m

# Default target
all: build

# Display help information
help:
	@echo "$(COLOR_BOLD)Wallpaper Generator - Build System$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_GREEN)Commands:$(COLOR_RESET)"
	@echo "  make build          - Build standalone Linux binary"
	@echo "  make dist           - Create distributable package (.tar.gz)"
	@echo "  make clean          - Remove build artifacts"
	@echo "  make clean-all      - Remove everything including venv"
	@echo "  make help           - Show this help message"
	@echo ""
	@echo "$(COLOR_YELLOW)Quick Start:$(COLOR_RESET)"
	@echo "  make build          # Creates dist/wallpaper-generator"
	@echo "  ./dist/wallpaper-generator"

# Create virtual environment and install dependencies
setup: $(VENV)/bin/activate

$(VENV)/bin/activate:
	@echo "$(COLOR_BLUE)Setting up build environment...$(COLOR_RESET)"
	$(PYTHON) -m venv $(VENV)
	$(PIP_VENV) install --upgrade pip setuptools wheel
	$(PIP_VENV) install pyinstaller
	@echo "$(COLOR_GREEN)✓ Build environment ready$(COLOR_RESET)"

# Build standalone binary
build: setup
	@echo "$(COLOR_BLUE)Building standalone binary for Linux...$(COLOR_RESET)"
	@echo ""
	$(BIN)/pyinstaller --clean --noconfirm \
		--name=$(BINARY_NAME) \
		--onefile \
		--add-data="llm.json:." \
		--add-data="config.py:." \
		--hidden-import=requests \
		--hidden-import=PIL \
		--hidden-import=dotenv \
		$(MAIN_SCRIPT)
	@echo ""
	@echo "$(COLOR_GREEN)✓ Binary built successfully!$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)Binary location: $(DIST_DIR)/$(BINARY_NAME)$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)Size: $$(du -h $(DIST_DIR)/$(BINARY_NAME) | cut -f1)$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_BLUE)To run:$(COLOR_RESET) ./$(DIST_DIR)/$(BINARY_NAME)"

# Create distributable package
dist: build
	@echo "$(COLOR_BLUE)Creating distributable package...$(COLOR_RESET)"
	@mkdir -p $(DIST_DIR)/package
	@cp $(DIST_DIR)/$(BINARY_NAME) $(DIST_DIR)/package/
	@if [ -f "llm.json" ]; then cp llm.json $(DIST_DIR)/package/; fi
	@if [ -f "README.md" ]; then cp README.md $(DIST_DIR)/package/; fi
	@echo "#!/bin/bash" > $(DIST_DIR)/package/install.sh
	@echo "cp $(BINARY_NAME) /usr/local/bin/" >> $(DIST_DIR)/package/install.sh
	@echo "chmod +x /usr/local/bin/$(BINARY_NAME)" >> $(DIST_DIR)/package/install.sh
	@echo "echo 'Installation complete! Run: $(BINARY_NAME)'" >> $(DIST_DIR)/package/install.sh
	@chmod +x $(DIST_DIR)/package/install.sh
	cd $(DIST_DIR) && tar -czf $(BINARY_NAME)-linux-x64.tar.gz package/
	@echo "$(COLOR_GREEN)✓ Package created!$(COLOR_RESET)"
	@echo "$(COLOR_YELLOW)Package: $(DIST_DIR)/$(BINARY_NAME)-linux-x64.tar.gz$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_BLUE)Distribution:$(COLOR_RESET)"
	@echo "  1. Share: $(DIST_DIR)/$(BINARY_NAME)-linux-x64.tar.gz"
	@echo "  2. Extract: tar -xzf $(BINARY_NAME)-linux-x64.tar.gz"
	@echo "  3. Run: ./package/$(BINARY_NAME)"
	@echo "  4. Install (optional): sudo ./package/install.sh"

# Clean build artifacts
clean:
	@echo "$(COLOR_BLUE)Removing build artifacts...$(COLOR_RESET)"
	rm -rf $(BUILD_DIR) $(DIST_DIR) $(SPEC_FILE)
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(COLOR_GREEN)✓ Build artifacts removed$(COLOR_RESET)"

# Clean everything including venv
clean-all: clean
	@echo "$(COLOR_BLUE)Removing virtual environment...$(COLOR_RESET)"
	rm -rf $(VENV)
	@echo "$(COLOR_GREEN)✓ Complete cleanup finished$(COLOR_RESET)"
