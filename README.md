# Miners Haven Miner

Our own RandomX CPU miner for [theminershaven.org](https://theminershaven.org), written in Rust.

- Coins: **QRL** and **Monero (XMR)**. GPU coins and a desktop app come next.
- Picks the right Miners Haven pool and port for you; SOLO or PPLNS.
- Fast mode (2 GB dataset, built on all cores, huge pages when available) or light mode (256 MB).
- Built-in self-test against the official RandomX test vectors, and a benchmark.

The RandomX algorithm itself is the official reference library by tevador (BSD-3-Clause), compiled in
through the `randomx-rs` crate. Everything else (pool protocol, job handling, threading, stats) is ours.

## Use

```
mhminer --coin qrl --wallet <your QRL address> --worker rig1
mhminer --coin xmr --wallet <your XMR address> --solo
mhminer --selftest          # check this machine computes correct hashes
mhminer --bench 30          # measure your hashrate, no pool needed
```

Options: `--threads N`, `--light`, `--large` (high-difficulty port for big CPUs / rentals),
`--pool host:port` (repeat for failover).

## Build

Rust 1.85+, CMake and a C++ compiler:

```
cargo build --release
```

Windows .exe from Linux (mingw-w64):

```
mkdir -p win-static && cp $(x86_64-w64-mingw32-g++ -print-file-name=libstdc++.a) win-static/libstdc++.dll.a
cargo build --release --target x86_64-pc-windows-gnu
```
