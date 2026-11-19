# Miners Haven Miner

One miner for CPU and GPU coins, made for [theminershaven.org](https://theminershaven.org) and written in Rust.
Windows desktop app plus the `mhminer` command line for Windows and Linux.

**0% dev fee on Miners Haven pools.**

[Download the latest release](https://github.com/minershaven/miners-haven-miner/releases/latest) ·
[Website](https://theminershaven.org) · [Discord](https://discord.gg/qwyVDDS9J)

## Coins

| Coin | Algorithm | Hardware |
| --- | --- | --- |
| Pearl (PRL) | Pearl | NVIDIA RTX 30-series or newer |
| Quantus (QTC) | QPoW | GPU or CPU |
| Neoxa (NEOX) | KawPoW | GPU |
| ABRS | KawPoW | GPU |
| Firo (FIRO) | FiroPoW | GPU |
| Zano (ZANO) | ProgPowZ | GPU |
| QRL | RandomX | CPU |
| Monero (XMR) | RandomX | CPU |
| DigiByte (DGB) | Skein | GPU |
| DigiByte (DGB) | Qubit | GPU |

GPU coins need the normal NVIDIA or AMD graphics driver (it includes OpenCL).

## Dev fee

- Miners Haven pools: **none**.
- Any other pool: 1.5% of the time (90 seconds in every 100 minutes, the first after 15 minutes).
  The log shows when the dev fee starts and ends, and shares found in that time are not counted in your totals.

## Features

- Desktop app: pick a coin, paste your wallet, press Start. Settings are remembered.
- Picks the right Miners Haven pool and port for you; SOLO or PPLNS.
- Region: auto (fastest), US (St. Louis) or EU (Germany), with the other server as backup.
- Per-GPU speed, temperature, fan and power.
- Huge pages for RandomX (about 10-20% faster).
- Mines on other pools too: choose the algorithm and fill in pool, wallet and password.

## Command line

```
mhminer --coin neox --wallet YOUR_ADDRESS --worker rig1
mhminer --coin qrl --wallet YOUR_ADDRESS --solo
mhminer --coin prl --bench 20        # Pearl speed test, NVIDIA RTX
mhminer --coin dgb-skein --wallet YOUR_DGB_ADDRESS --worker rig1
mhminer --coin xmr --bench 20        # CPU speed test
```

| Option | Meaning |
| --- | --- |
| `--coin X` | qrl, xmr, qtc, neox, abrs, firo, zano, prl, dgb-skein, dgb-qubit |
| `--solo` | mine SOLO instead of PPLNS |
| `--region X` | auto, us or eu (default auto) |
| `--large` | high-difficulty port for big rigs and rentals |
| `--threads N` | CPU threads (CPU coins) |
| `--gpu N` | graphics card to use (repeat for more) |
| `--pool HOST:PORT` | another pool (`stratum+ssl://HOST:PORT` for SSL) |
| `--user` / `--pass` | login and password for other pools |

Windows Defender may warn about a crypto miner; that is expected for an unsigned miner.

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

RandomX uses the official reference library by tevador (BSD-3-Clause) through the `randomx-rs` crate.
The DigiByte Skein and Qubit hashes (CPU and OpenCL) are built from sphlib by Thomas Pornin, Projet RNRT SAPHIR (MIT).
