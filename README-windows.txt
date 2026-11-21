Miners Haven Miner 2.8  -  theminershaven.org
=============================================

COINS
  On your processor (CPU):    QRL (RandomX)
  On your graphics card (GPU): ABRS (KawPoW)  FIRO (FiroPoW)  ZANO (ProgPowZ)
  On your graphics card (GPU): ERG (Ergo, Autolykos2) - needs 8 GB of graphics memory or more
  On GPU and/or CPU:           QTC (QPoW)
  On your graphics card (GPU): DGB (DigiByte Skein or Qubit)
  On an NVIDIA RTX card:       PRL (Pearl) - RTX 30 series or newer; much slower on a CPU
  GPU coins need the normal NVIDIA or AMD graphics driver (it includes OpenCL).

DESKTOP APP
  Double-click MinersHavenMiner.exe. Pick a coin, paste your wallet address and press
  START MINING. For GPU coins, tick the graphics cards to use. Settings are remembered.
  Pool: "Miners Haven" picks the right port for you. "Other pool" mines anywhere: choose the
  algorithm, then fill in the pool address, your wallet (or username) and the password.

DEV FEE
  Mining on Miners Haven pools is free: no dev fee at all.
  On any other pool the miner works for the developer for 1.5% of the time (90 seconds
  in every 100 minutes, the first after 15 minutes). The log shows "dev fee" lines when
  it starts and ends, and shares found in that time are not counted in your totals.
  Location: Auto (fastest), US (St. Louis) or EU (Germany); the other server is the backup.
  GPU coins show each card's speed, temperature, fan and power. The processor temperature is shown
  for every coin (on Windows it comes from the motherboard's thermal sensor; some boards don't report one).

  Huge pages (RandomX only, about 10-20% faster): right-click MinersHavenMiner.exe >
  Run as administrator (or press "Restart as administrator" in the app). The first time,
  sign out of Windows and back in (or restart) once afterwards.

COMMAND LINE (mhminer.exe)
  mhminer.exe --coin erg --wallet YOUR_ERGO_ADDRESS --worker rig1
  mhminer.exe --coin zano --bench 20        (GPU speed test)
  mhminer.exe --coin prl --bench 20         (Pearl speed test, NVIDIA RTX)
  mhminer.exe --coin dgb-skein --wallet YOUR_DGB_ADDRESS --worker rig1
  mhminer.exe --coin qrl --bench 20         (CPU speed test)

  Options:
    --coin X      qrl, qtc, abrs, firo, zano, prl, dgb-skein, dgb-qubit, erg
    --solo        mine SOLO instead of PPLNS
    --region X    auto, us or eu (default auto: the faster server, other one as backup)
    --large       high-difficulty port for big rigs / rentals
    --threads 4   CPU threads (CPU coins)
    --gpu 0       which graphics card(s) to use (GPU coins; repeat for more)
    --pool HOST:PORT   another pool (stratum+ssl://HOST:PORT for SSL)
    --user / --pass    login and password for other pools

Windows Defender may warn about a crypto miner; that is expected for an unsigned miner.
