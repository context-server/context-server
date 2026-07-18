# Build Linux wheels for context-server (local podman/docker helper).
#
# ORT's prebuilt static libraries need glibc >= ~2.38 (__isoc23_*,
# __libc_single_threaded), so classic manylinux_2_28 images cannot link.
# Ubuntu 24.04 (glibc 2.39) matches that requirement.
#
# GitHub Actions builds wheels natively on ubuntu-24.04 with rust-cache instead
# of this image (see .github/workflows/ci.yml / release.yml).
#
# Local:
#   ./scripts/build-wheel.sh
#   VERSION=2026.716.1 ./scripts/build-wheel.sh
#
# Extract without the helper:
#   podman build --build-arg VERSION=2026.716.1 -t context-server-wheel -f Containerfile .
#   cid=$(podman create context-server-wheel)
#   podman cp "$cid:/out/." ./dist/ && podman rm "$cid"

FROM docker.io/library/ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH \
    OPENSSL_NO_VENDOR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        libssl-dev \
        patchelf \
        pkg-config \
        python3 \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --default-toolchain stable --profile minimal \
    && rustc --version \
    && cargo --version

RUN python3 -m pip install --break-system-packages 'maturin[patchelf]>=1.0,<2.0' \
    && maturin --version

WORKDIR /src
COPY . .

# Optional CalVer override (YYYY.MMDD.N) from CI / build-wheel.sh
ARG VERSION=
RUN if [ -n "$VERSION" ]; then bash ./scripts/set-version.sh "$VERSION"; fi

# Tag as manylinux_2_39; maturin/patchelf vendors libssl into the wheel for PyPI.
RUN mkdir -p /out \
    && maturin build --release --locked --compatibility manylinux_2_39 -o /out \
    && maturin sdist -o /out \
    && ls -la /out

CMD ["ls", "-la", "/out"]
