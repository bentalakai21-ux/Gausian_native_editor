use crate::{PluginContext, PluginError, PluginResult};
use anyhow::Result;
use wasmtime::*;

/// WASM runtime for executing WebAssembly plugins
pub struct WasmRuntime {
    engine: Engine,
    linker: Linker<WasmPluginState>,
}

/// State passed to WASM plugin instances
pub struct WasmPluginState {
    pub context: PluginContext,
    pub logs: Vec<String>,
    pub memory_limit: usize,
}

impl WasmRuntime {
    pub fn new() -> Result<Self> {
        let mut config = Config::new();
        config.wasm_simd(true);
        config.wasm_bulk_memory(true);
        config.wasm_multi_value(true);
        config.consume_fuel(true);

        let engine = Engine::new(&config)?;
        let mut linker = Linker::new(&engine);

        // Add WASI support would go here
        // wasmtime_wasi::add_to_linker(&mut linker, |state: &mut WasmPluginState| {
        //     // Create WASI context with limited permissions
        //     &mut state.context
        // })?;

        // Add custom host functions
        Self::define_host_functions(&mut linker)?;

        Ok(Self { engine, linker })
    }

    pub fn execute_plugin(&self, module: &Module, context: PluginContext) -> Result<PluginResult> {
        let mut store = Store::new(
            &self.engine,
            WasmPluginState {
                context,
                logs: Vec::new(),
                memory_limit: 64 * 1024 * 1024, // 64MB limit
            },
        );

        // Set fuel for execution limits
        store.set_fuel(10_000_000)?; // Limit execution time

        let instance = self.linker.instantiate(&mut store, module)?;

        // Call the plugin's main function
        let main_func = instance
            .get_typed_func::<(), i32>(&mut store, "plugin_main")
            .map_err(|_| PluginError::WasmRuntime("plugin_main function not found".to_string()))?;

        let result_code = main_func.call(&mut store, ())?;

        let state = store.data();
        let success = result_code == 0;

        Ok(PluginResult {
            success,
            output_items: vec![], // TODO: Extract from WASM memory
            modified_sequence: None,
            artifacts: vec![],
            logs: state.logs.clone(),
            error_message: if success {
                None
            } else {
                Some("Plugin returned error code".to_string())
            },
        })
    }

    fn define_host_functions(linker: &mut Linker<WasmPluginState>) -> Result<()> {
        // Log function
        linker.func_wrap(
            "env",
            "log",
            |mut caller: Caller<'_, WasmPluginState>,
             ptr: i32,
             len: i32|
             -> Result<(), anyhow::Error> {
                let memory = caller
                    .get_export("memory")
                    .and_then(|e| e.into_memory())
                    .ok_or_else(|| anyhow::anyhow!("Failed to get memory"))?;

                let log_str = {
                    let data = memory.data(&caller);
                    let log_bytes = &data[ptr as usize..(ptr + len) as usize];
                    String::from_utf8_lossy(log_bytes).to_string()
                };

                caller.data_mut().logs.push(log_str);
                Ok(())
            },
        )?;

        // Get current frame
        linker.func_wrap(
            "env",
            "get_current_frame",
            |caller: Caller<'_, WasmPluginState>| caller.data().context.current_frame,
        )?;

        // Get sequence width
        linker.func_wrap("env", "get_width", |caller: Caller<'_, WasmPluginState>| {
            caller.data().context.width as i32
        })?;

        // Get sequence height
        linker.func_wrap(
            "env",
            "get_height",
            |caller: Caller<'_, WasmPluginState>| caller.data().context.height as i32,
        )?;

        // Memory allocation (simple bump allocator)
        linker.func_wrap(
            "env",
            "alloc",
            |mut _caller: Caller<'_, WasmPluginState>, _size: i32| -> i32 {
                // Simple allocation - in a real implementation, you'd want a proper allocator
                // For now, just return a fixed offset (this is a stub)
                // In practice, you'd implement a proper WASM allocator
                1024 // Return offset in WASM memory
            },
        )?;

        Ok(())
    }
}

/// Helper functions for WASM plugin development
pub mod wasm_helpers {
    /// Generate a basic WASM plugin template in Rust
    pub fn generate_wasm_plugin_template(plugin_name: &str) -> String {
        format!(
            r#"//! {plugin_name} - A WASM plugin for Gausian Native Editor

use std::ffi::CStr;
use std::os::raw::{{c_char, c_int}};

// Host function bindings
extern "C" {{
    fn log(ptr: *const c_char, len: c_int);
    fn get_current_frame() -> i64;
    fn get_width() -> c_int;
    fn get_height() -> c_int;
    fn alloc(size: c_int) -> *mut u8;
}}

// Helper function to log messages
fn plugin_log(message: &str) {{
    unsafe {{
        log(message.as_ptr() as *const c_char, message.len() as c_int);
    }}
}}

// Main plugin entry point
#[no_mangle]
pub extern "C" fn plugin_main() -> c_int {{
    plugin_log("Plugin {plugin_name} starting");
    
    let current_frame = unsafe {{ get_current_frame() }};
    let width = unsafe {{ get_width() }};
    let height = unsafe {{ get_height() }};
    
    plugin_log(&format!("Processing frame {{}} ({{}}x{{}})", current_frame, width, height));
    
    // TODO: Implement your plugin logic here
    
    plugin_log("Plugin {plugin_name} completed successfully");
    
    0 // Return 0 for success
}}

// Plugin metadata
#[no_mangle]
pub extern "C" fn plugin_get_name() -> *const c_char {{
    "{plugin_name}\0".as_ptr() as *const c_char
}}

#[no_mangle]
pub extern "C" fn plugin_get_version() -> *const c_char {{
    "1.0.0\0".as_ptr() as *const c_char
}}
"#,
            plugin_name = plugin_name
        )
    }

    /// Generate Cargo.toml for a WASM plugin
    pub fn generate_wasm_cargo_toml(plugin_name: &str) -> String {
        format!(
            r#"[package]
name = "{plugin_name}"
version = "1.0.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
# Add your dependencies here

[profile.release]
lto = true
opt-level = "s"  # Optimize for size
panic = "abort"
"#,
            plugin_name = plugin_name.to_lowercase().replace(' ', "-")
        )
    }
}
