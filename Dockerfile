FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

RUN apt-get update && apt-get install -y \
    git cmake build-essential curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN git clone https://github.com/ggml-org/llama.cpp.git

WORKDIR /build/llama.cpp

ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LD_LIBRARY_PATH

RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda" \
    && cmake --build build --target llama-server --config Release -j2


FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    curl ca-certificates python3 python3-pip \
    && pip3 install --no-cache-dir -U "huggingface_hub[cli]" \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /build/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=builder /build/llama.cpp/build/bin/*.so* /usr/local/lib/

RUN ldconfig

RUN mkdir -p /models/medgemma

RUN hf download \
    unsloth/medgemma-4b-it-GGUF \
    medgemma-4b-it-Q8_0.gguf \
    --local-dir /models/medgemma

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

EXPOSE 8000

ENV MODEL_PATH=/models/medgemma/medgemma-4b-it-Q8_0.gguf
ENV CTX_SIZE=4096
ENV N_GPU_LAYERS=999

CMD ["/app/start.sh"]