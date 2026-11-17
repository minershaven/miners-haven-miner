// Embed the app icon and version info into the Windows .exe files.
fn main() {
    println!("cargo:rerun-if-changed=assets/icon.ico");
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let ver = std::env::var("CARGO_PKG_VERSION").unwrap();
    let nums = ver.replace('.', ",") + ",0";
    let rc = format!(r#"1 ICON "{icon}"
1 VERSIONINFO
FILEVERSION {nums}
PRODUCTVERSION {nums}
BEGIN
  BLOCK "StringFileInfo"
  BEGIN
    BLOCK "040904B0"
    BEGIN
      VALUE "CompanyName", "Miners Haven"
      VALUE "FileDescription", "Miners Haven Miner"
      VALUE "FileVersion", "{ver}"
      VALUE "ProductName", "Miners Haven Miner"
      VALUE "ProductVersion", "{ver}"
      VALUE "LegalCopyright", "theminershaven.org"
    END
  END
  BLOCK "VarFileInfo"
  BEGIN
    VALUE "Translation", 0x409, 1200
  END
END
"#, icon = std::env::current_dir().unwrap().join("assets/icon.ico").display());
    let rc_path = out.join("app.rc");
    std::fs::write(&rc_path, rc).unwrap();
    let obj = out.join("app-res.o");
    let windres = std::env::var("WINDRES").unwrap_or_else(|_| "x86_64-w64-mingw32-windres".into());
    let ok = std::process::Command::new(&windres).arg(&rc_path).arg("-O").arg("coff").arg("-o").arg(&obj).status()
        .map(|s| s.success()).unwrap_or(false);
    if ok {
        println!("cargo:rustc-link-arg-bins={}", obj.display());
    } else {
        println!("cargo:warning=windres failed; building without an icon");
    }
}
