Miners Haven Miner 0.1  -  theminershaven.org
=============================================

DESKTOP APP
  Double-click MinersHavenMiner.exe. Pick QRL or Monero, paste your wallet address,
  choose how many CPU threads to use and press START MINING. Settings are remembered.

  Huge pages (about 10-20% faster): right-click MinersHavenMiner.exe > Run as administrator
  (or press "Restart as administrator" in the app). The first time, it gives your Windows
  account the "Lock pages in memory" permission; sign out of Windows and back in (or restart)
  once, then run it as administrator again and "Huge pages" shows On.

COMMAND LINE (mhminer.exe)
  1. Double-click SELFTEST-AND-BENCHMARK.bat to check your PC and see your hashrate.
  2. Right-click START-QRL.bat (or START-XMR.bat) > Edit, replace YOUR_QRL_ADDRESS with your
     wallet address, save, then double-click it to start mining.

  Options (add them to the line in the .bat file):
    --solo        mine SOLO instead of PPLNS
    --threads 4   use a set number of CPU threads
    --light       use 256 MB of RAM instead of 2 GB (slower)
    --large       high-difficulty port for big CPUs / rentals

Windows Defender may warn about a crypto miner; that is expected for an unsigned miner.
