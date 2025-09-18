fn main() {
    #[cfg(target_os = "macos")]
    {
        // Link system frameworks
        println!("cargo:rustc-link-lib=framework=VideoToolbox");
        println!("cargo:rustc-link-lib=framework=CoreVideo");
        println!("cargo:rustc-link-lib=framework=CoreMedia");
        println!("cargo:rustc-link-lib=framework=AVFoundation");
        println!("cargo:rustc-link-lib=framework=CoreFoundation");
        println!("cargo:rustc-link-lib=framework=IOSurface");
        // Align deployment target with the top-level linker flags
        println!("cargo:rustc-env=MACOSX_DEPLOYMENT_TARGET=11.0");

        // Compile the Obj-C shim
        let shim_path = std::path::Path::new("src/avfoundation_shim.m");
        if shim_path.exists() {
            let mut build = cc::Build::new();
            build
                .file("src/avfoundation_shim.m")
                .flag("-fobjc-arc")
                .flag("-mmacosx-version-min=11.0")
                .compile("avfoundation_shim");
        }
        // Rebuild shim when these files change
        println!("cargo:rerun-if-changed=src/avfoundation_shim.m");
        println!("cargo:rerun-if-changed=src/avfoundation_shim.h");
    }
}
