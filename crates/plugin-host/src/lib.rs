use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use thiserror::Error;
use timeline::{Item, ItemKind, Sequence};
use tokio::sync::{mpsc, RwLock};
use uuid::Uuid;
use wasmtime::*;

pub mod marketplace;
pub mod python_bridge;
pub mod wasm_runtime;

#[derive(Debug, Error)]
pub enum PluginError {
    #[error("Plugin not found: {0}")]
    NotFound(String),
    #[error("Plugin load error: {0}")]
    LoadError(String),
    #[error("Plugin execution error: {0}")]
    ExecutionError(String),
    #[error("Invalid plugin manifest: {0}")]
    InvalidManifest(String),
    #[error("Python bridge error: {0}")]
    PythonBridge(String),
    #[error("WASM runtime error: {0}")]
    WasmRuntime(String),
    #[error("Timeout: {0}")]
    Timeout(String),
    #[error("Security violation: {0}")]
    SecurityViolation(String),
}

/// Plugin manifest structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginManifest {
    pub name: String,
    pub version: String,
    pub author: String,
    pub description: String,
    pub plugin_type: PluginType,
    pub runtime: PluginRuntime,
    pub entry_point: String,
    pub capabilities: Vec<PluginCapability>,
    pub parameters: Vec<PluginParameter>,
    pub signature: Option<String>,
    pub dependencies: Vec<String>,
    pub minimum_host_version: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginType {
    Effect,
    Generator,
    Transition,
    AudioProcessor,
    ColorCorrection,
    Stabilization,
    AiWorkflow,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginRuntime {
    Wasm,
    Python,
    Native,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PluginCapability {
    ProcessVideo,
    ProcessAudio,
    GenerateContent,
    AccessNetwork,
    AccessFileSystem,
    GpuAcceleration,
    MultiThreading,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginParameter {
    pub name: String,
    pub display_name: String,
    pub param_type: ParameterType,
    pub default_value: serde_json::Value,
    pub min_value: Option<serde_json::Value>,
    pub max_value: Option<serde_json::Value>,
    pub description: String,
    pub group: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterType {
    Float,
    Integer,
    Boolean,
    String,
    Color,
    File,
    Enum { options: Vec<String> },
    Range { min: f64, max: f64 },
}

/// Plugin execution context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginContext {
    pub sequence: Sequence,
    pub current_frame: i64,
    pub fps: f32,
    pub width: u32,
    pub height: u32,
    pub parameters: HashMap<String, serde_json::Value>,
    pub temp_dir: PathBuf,
    pub project_dir: Option<PathBuf>,
}

/// Plugin execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginResult {
    pub success: bool,
    pub output_items: Vec<Item>,
    pub modified_sequence: Option<Sequence>,
    pub artifacts: Vec<PluginArtifact>,
    pub logs: Vec<String>,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginArtifact {
    pub name: String,
    pub path: PathBuf,
    pub artifact_type: ArtifactType,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArtifactType {
    Video,
    Audio,
    Image,
    Data,
    Log,
}

/// Main plugin host
pub struct PluginHost {
    plugins: Arc<RwLock<HashMap<String, LoadedPlugin>>>,
    wasm_engine: Engine,
    python_bridge: Option<python_bridge::PythonBridge>,
    security_policy: SecurityPolicy,
    resource_limits: ResourceLimits,
}

#[derive(Debug)]
pub struct LoadedPlugin {
    pub manifest: PluginManifest,
    pub plugin_path: PathBuf,
    pub runtime_handle: PluginRuntimeHandle,
    pub last_used: std::time::Instant,
}

#[derive(Debug)]
pub enum PluginRuntimeHandle {
    Wasm(Module),
    Python(PathBuf),
    Native(libloading::Library),
}

#[derive(Debug, Clone)]
pub struct SecurityPolicy {
    pub allow_network_access: bool,
    pub allow_file_system_access: bool,
    pub allowed_directories: Vec<PathBuf>,
    pub max_memory_mb: u64,
    pub max_execution_time_sec: u64,
    pub require_signature: bool,
}

#[derive(Debug, Clone)]
pub struct ResourceLimits {
    pub max_memory_bytes: u64,
    pub max_cpu_time_ms: u64,
    pub max_temp_files: u32,
    pub max_temp_size_bytes: u64,
}

impl Default for SecurityPolicy {
    fn default() -> Self {
        Self {
            allow_network_access: false,
            allow_file_system_access: false,
            allowed_directories: vec![],
            max_memory_mb: 512,
            max_execution_time_sec: 30,
            require_signature: true,
        }
    }
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_bytes: 512 * 1024 * 1024, // 512 MB
            max_cpu_time_ms: 30_000,             // 30 seconds
            max_temp_files: 100,
            max_temp_size_bytes: 1024 * 1024 * 1024, // 1 GB
        }
    }
}

impl PluginHost {
    pub fn new() -> Result<Self> {
        let wasm_config = Config::new();
        let wasm_engine = Engine::new(&wasm_config)?;

        Ok(Self {
            plugins: Arc::new(RwLock::new(HashMap::new())),
            wasm_engine,
            python_bridge: None,
            security_policy: SecurityPolicy::default(),
            resource_limits: ResourceLimits::default(),
        })
    }

    pub fn with_python_bridge(mut self) -> Result<Self> {
        self.python_bridge = Some(python_bridge::PythonBridge::new()?);
        Ok(self)
    }

    pub fn set_security_policy(&mut self, policy: SecurityPolicy) {
        self.security_policy = policy;
    }

    pub fn set_resource_limits(&mut self, limits: ResourceLimits) {
        self.resource_limits = limits;
    }

    /// Load a plugin from a directory
    pub async fn load_plugin(&self, plugin_dir: &Path) -> Result<String> {
        let manifest_path = plugin_dir.join("plugin.json");
        let manifest_content = tokio::fs::read_to_string(&manifest_path).await?;
        let manifest: PluginManifest = serde_json::from_str(&manifest_content)
            .map_err(|e| PluginError::InvalidManifest(e.to_string()))?;

        // Validate plugin signature if required
        if self.security_policy.require_signature {
            self.validate_plugin_signature(&manifest, plugin_dir)?;
        }

        // Load the plugin based on its runtime
        let runtime_handle = match manifest.runtime {
            PluginRuntime::Wasm => {
                let wasm_path = plugin_dir.join(&manifest.entry_point);
                let wasm_bytes = tokio::fs::read(&wasm_path).await?;
                let module = Module::new(&self.wasm_engine, &wasm_bytes)
                    .map_err(|e| PluginError::WasmRuntime(e.to_string()))?;
                PluginRuntimeHandle::Wasm(module)
            }
            PluginRuntime::Python => {
                let python_path = plugin_dir.join(&manifest.entry_point);
                if !python_path.exists() {
                    return Err(
                        PluginError::LoadError("Python entry point not found".to_string()).into(),
                    );
                }
                PluginRuntimeHandle::Python(python_path)
            }
            PluginRuntime::Native => {
                return Err(
                    PluginError::LoadError("Native plugins not yet supported".to_string()).into(),
                );
            }
        };

        let plugin_id = format!("{}:{}", manifest.name, manifest.version);
        let loaded_plugin = LoadedPlugin {
            manifest,
            plugin_path: plugin_dir.to_path_buf(),
            runtime_handle,
            last_used: std::time::Instant::now(),
        };

        let mut plugins = self.plugins.write().await;
        plugins.insert(plugin_id.clone(), loaded_plugin);

        tracing::info!("Loaded plugin: {}", plugin_id);
        Ok(plugin_id)
    }

    /// Execute a plugin with given context
    pub async fn execute_plugin(
        &self,
        plugin_id: &str,
        context: PluginContext,
    ) -> Result<PluginResult> {
        let plugins = self.plugins.read().await;
        let plugin = plugins
            .get(plugin_id)
            .ok_or_else(|| PluginError::NotFound(plugin_id.to_string()))?;

        // Create isolated execution environment
        let temp_dir = tempfile::tempdir()?;
        let mut execution_context = context;
        execution_context.temp_dir = temp_dir.path().to_path_buf();

        // Execute based on runtime
        match &plugin.runtime_handle {
            PluginRuntimeHandle::Wasm(module) => {
                self.execute_wasm_plugin(module, &plugin.manifest, execution_context)
                    .await
            }
            PluginRuntimeHandle::Python(script_path) => {
                if let Some(ref bridge) = self.python_bridge {
                    bridge
                        .execute_plugin(script_path, &plugin.manifest, execution_context)
                        .await
                } else {
                    Err(
                        PluginError::PythonBridge("Python bridge not initialized".to_string())
                            .into(),
                    )
                }
            }
            PluginRuntimeHandle::Native(_) => Err(PluginError::ExecutionError(
                "Native plugins not yet supported".to_string(),
            )
            .into()),
        }
    }

    /// List all loaded plugins
    pub async fn list_plugins(&self) -> Vec<String> {
        let plugins = self.plugins.read().await;
        plugins.keys().cloned().collect()
    }

    /// Get plugin manifest
    pub async fn get_plugin_manifest(&self, plugin_id: &str) -> Option<PluginManifest> {
        let plugins = self.plugins.read().await;
        plugins.get(plugin_id).map(|p| p.manifest.clone())
    }

    /// Unload a plugin
    pub async fn unload_plugin(&self, plugin_id: &str) -> Result<()> {
        let mut plugins = self.plugins.write().await;
        plugins
            .remove(plugin_id)
            .ok_or_else(|| PluginError::NotFound(plugin_id.to_string()))?;

        tracing::info!("Unloaded plugin: {}", plugin_id);
        Ok(())
    }

    /// Scan directory for plugins and load them
    pub async fn scan_and_load_plugins(&self, plugins_dir: &Path) -> Result<Vec<String>> {
        let mut loaded_plugins = Vec::new();

        if !plugins_dir.exists() {
            return Ok(loaded_plugins);
        }

        let mut entries = tokio::fs::read_dir(plugins_dir).await?;
        while let Some(entry) = entries.next_entry().await? {
            if entry.file_type().await?.is_dir() {
                match self.load_plugin(&entry.path()).await {
                    Ok(plugin_id) => loaded_plugins.push(plugin_id),
                    Err(e) => {
                        tracing::warn!("Failed to load plugin from {:?}: {}", entry.path(), e)
                    }
                }
            }
        }

        Ok(loaded_plugins)
    }

    fn validate_plugin_signature(
        &self,
        _manifest: &PluginManifest,
        _plugin_dir: &Path,
    ) -> Result<()> {
        // TODO: Implement plugin signature validation
        // For now, just return Ok if signature validation is required but not implemented
        tracing::warn!("Plugin signature validation not yet implemented");
        Ok(())
    }

    async fn execute_wasm_plugin(
        &self,
        _module: &Module,
        _manifest: &PluginManifest,
        _context: PluginContext,
    ) -> Result<PluginResult> {
        // TODO: Implement WASM plugin execution
        // This would involve creating a WASM instance, setting up WASI, and calling the plugin
        tracing::warn!("WASM plugin execution not yet fully implemented");

        Ok(PluginResult {
            success: true,
            output_items: vec![],
            modified_sequence: None,
            artifacts: vec![],
            logs: vec!["WASM plugin executed (stub)".to_string()],
            error_message: None,
        })
    }
}

impl Default for PluginHost {
    fn default() -> Self {
        Self::new().expect("Failed to create default PluginHost")
    }
}

/// Utility functions for plugin development
pub mod utils {
    use super::*;

    pub fn create_plugin_manifest(
        name: &str,
        version: &str,
        author: &str,
        plugin_type: PluginType,
        runtime: PluginRuntime,
        entry_point: &str,
    ) -> PluginManifest {
        PluginManifest {
            name: name.to_string(),
            version: version.to_string(),
            author: author.to_string(),
            description: String::new(),
            plugin_type,
            runtime,
            entry_point: entry_point.to_string(),
            capabilities: vec![],
            parameters: vec![],
            signature: None,
            dependencies: vec![],
            minimum_host_version: "0.1.0".to_string(),
        }
    }

    pub fn create_float_parameter(
        name: &str,
        display_name: &str,
        default: f64,
        min: Option<f64>,
        max: Option<f64>,
        description: &str,
    ) -> PluginParameter {
        PluginParameter {
            name: name.to_string(),
            display_name: display_name.to_string(),
            param_type: ParameterType::Float,
            default_value: serde_json::json!(default),
            min_value: min.map(|v| serde_json::json!(v)),
            max_value: max.map(|v| serde_json::json!(v)),
            description: description.to_string(),
            group: None,
        }
    }

    pub fn create_enum_parameter(
        name: &str,
        display_name: &str,
        options: Vec<String>,
        default_index: usize,
        description: &str,
    ) -> PluginParameter {
        let default_value = options.get(default_index).cloned().unwrap_or_default();

        PluginParameter {
            name: name.to_string(),
            display_name: display_name.to_string(),
            param_type: ParameterType::Enum { options },
            default_value: serde_json::json!(default_value),
            min_value: None,
            max_value: None,
            description: description.to_string(),
            group: None,
        }
    }
}
