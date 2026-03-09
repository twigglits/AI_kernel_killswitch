# CUDA LLM Inference Engine
# Target: RTX 4070 SUPER (sm_89)

NVCC      := nvcc
NVCCFLAGS := -arch=sm_89 -O2 -std=c++17 --expt-relaxed-constexpr
LDFLAGS   := -lcublas -lcuda

SRC_DIR   := src
BUILD_DIR := build

# Shared object files
SHARED_OBJS := $(BUILD_DIR)/cublas_ops.o $(BUILD_DIR)/sampler.o

# ---- GPT-2 target ----
GPT2_TARGET := $(BUILD_DIR)/gpt2_inference
GPT2_OBJS   := $(BUILD_DIR)/kernels.o $(BUILD_DIR)/tokenizer.o \
               $(BUILD_DIR)/model.o $(BUILD_DIR)/main.o

# ---- Llama target ----
LLAMA_TARGET := $(BUILD_DIR)/llama_inference
LLAMA_OBJS   := $(BUILD_DIR)/llama_kernels.o $(BUILD_DIR)/llama_tokenizer.o \
                $(BUILD_DIR)/llama_model.o $(BUILD_DIR)/llama_main.o

.PHONY: all gpt2 llama clean run run-llama download download-llama

all: gpt2 llama

gpt2: $(GPT2_TARGET)

llama: $(LLAMA_TARGET)

$(GPT2_TARGET): $(GPT2_OBJS) $(SHARED_OBJS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

$(LLAMA_TARGET): $(LLAMA_OBJS) $(SHARED_OBJS) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(SRC_DIR) -c -o $@ $<

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)

# ---- Run targets ----
run: $(GPT2_TARGET)
	./$(GPT2_TARGET) --prompt "The meaning of life is"

run-llama: $(LLAMA_TARGET)
	./$(LLAMA_TARGET) --chat --prompt "What is 3+2?"

# ---- Download targets ----
download:
	.venv/bin/python3 scripts/download_model.py

download-llama:
	.venv/bin/python3 scripts/download_llama.py
