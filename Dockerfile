# ARM64 explicit — prevents Docker from silently pulling x86 layers
FROM --platform=linux/arm64 ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    build-essential cmake git python3.12 python3.12-venv \
    python3-pip wget curl numactl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

# Python env
RUN python3.12 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip && pip install -r requirements.txt

# Build llama.cpp KleidiAI inside container
RUN git clone https://github.com/ggml-org/llama.cpp /opt/llama.cpp && \
    cmake -B /opt/llama.cpp/build_kleidiai \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_KLEIDIAI=ON \
        -DGGML_CPU_ARM_ARCH="armv8.2-a+dotprod+i8mm" \
        -DLLAMA_BUILD_SERVER_WEBUI=OFF \
        /opt/llama.cpp && \
    cmake --build /opt/llama.cpp/build_kleidiai --config Release -j$(nproc)

ENV LLAMA_CPP_HOME=/opt/llama.cpp
EXPOSE 8080

CMD ["uvicorn", "armforge.dashboard.app:app", "--host", "0.0.0.0", "--port", "8080"]
