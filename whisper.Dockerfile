FROM ubuntu:24.04 AS builder

RUN apt-get update && apt-get install -y \
    build-essential cmake git curl \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/ggml-org/whisper.cpp.git /whisper.cpp
WORKDIR /whisper.cpp

RUN cmake -B build \
      -DWHISPER_METAL=OFF \
      -DWHISPER_CUDA=OFF \
      -DWHISPER_BUILD_SERVER=ON \
      -DBUILD_SHARED_LIBS=ON \
    && cmake --build build --config Release -j$(nproc)

FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends curl libgomp1 ffmpeg && rm -rf /var/lib/apt/lists/*

COPY --from=builder /whisper.cpp/build/bin/whisper-server /usr/local/bin/
COPY --from=builder /whisper.cpp/build/src/libwhisper.so* /usr/local/lib/
COPY --from=builder /whisper.cpp/build/ggml/src/libggml*.so* /usr/local/lib/
RUN ldconfig

EXPOSE 8080
ENTRYPOINT ["whisper-server"]
CMD ["--host", "0.0.0.0", "--port", "8080", "-m", "/models/ggml-small.bin"]
