use crate::{PluginContext, PluginError, PluginManifest, PluginResult};
use anyhow::Result;
use serde_json;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command as TokioCommand};
use tokio::sync::mpsc;
use tokio::time::timeout;
use tracing::{debug, error, warn};

/// Python bridge for executing Python-based plugins
pub struct PythonBridge {
    python_executable: String,
    base_env: std::collections::HashMap<String, String>,
    max_execution_time: Duration,
}

impl PythonBridge {
    pub fn new() -> Result<Self> {
        // Try to find Python executable
        let python_executable = Self::find_python_executable()?;

        // Verify Python installation and required packages
        Self::verify_python_environment(&python_executable)?;

        Ok(Self {
            python_executable,
            base_env: std::env::vars().collect(),
            max_execution_time: Duration::from_secs(300), // 5 minutes default
        })
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.max_execution_time = timeout;
        self
    }

    /// Execute a Python plugin
    pub async fn execute_plugin(
        &self,
        script_path: &Path,
        manifest: &PluginManifest,
        context: PluginContext,
    ) -> Result<PluginResult> {
        debug!("Executing Python plugin: {:?}", script_path);

        // Prepare the execution environment
        let input_data = serde_json::to_string(&context)?;
        let temp_input = context.temp_dir.join("input.json");
        let temp_output = context.temp_dir.join("output.json");
        let temp_logs = context.temp_dir.join("logs.txt");

        tokio::fs::write(&temp_input, &input_data).await?;

        // Prepare Python command
        let mut cmd = TokioCommand::new(&self.python_executable);
        cmd.arg(script_path)
            .arg("--input")
            .arg(&temp_input)
            .arg("--output")
            .arg(&temp_output)
            .arg("--logs")
            .arg(&temp_logs)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .current_dir(&context.temp_dir);

        // Set environment variables
        for (key, value) in &self.base_env {
            cmd.env(key, value);
        }

        // Add plugin-specific environment
        cmd.env("PLUGIN_NAME", &manifest.name)
            .env("PLUGIN_VERSION", &manifest.version)
            .env("TEMP_DIR", &context.temp_dir)
            .env(
                "PYTHONPATH",
                script_path.parent().unwrap_or_else(|| Path::new(".")),
            );

        // Execute with timeout
        let execution_result = timeout(self.max_execution_time, async {
            let mut child = cmd.spawn()?;

            // Capture stdout and stderr
            let stdout = child.stdout.take().unwrap();
            let stderr = child.stderr.take().unwrap();

            let (stdout_tx, mut stdout_rx) = mpsc::channel(100);
            let (stderr_tx, mut stderr_rx) = mpsc::channel(100);

            // Spawn tasks to read stdout and stderr
            tokio::spawn(async move {
                let mut reader = BufReader::new(stdout);
                let mut line = String::new();
                while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
                    let _ = stdout_tx.send(line.trim().to_string()).await;
                    line.clear();
                }
            });

            tokio::spawn(async move {
                let mut reader = BufReader::new(stderr);
                let mut line = String::new();
                while reader.read_line(&mut line).await.unwrap_or(0) > 0 {
                    let _ = stderr_tx.send(line.trim().to_string()).await;
                    line.clear();
                }
            });

            let mut stdout_lines = Vec::new();
            let mut stderr_lines = Vec::new();

            // Collect output while process is running
            loop {
                tokio::select! {
                    Some(line) = stdout_rx.recv() => {
                        debug!("Plugin stdout: {}", line);
                        stdout_lines.push(line);
                    }
                    Some(line) = stderr_rx.recv() => {
                        warn!("Plugin stderr: {}", line);
                        stderr_lines.push(line);
                    }
                    status = child.wait() => {
                        match status {
                            Ok(exit_status) => {
                                if exit_status.success() {
                                    break Ok((stdout_lines, stderr_lines));
                                } else {
                                    break Err(PluginError::ExecutionError(
                                        format!("Plugin exited with code: {:?}", exit_status.code())
                                    ).into());
                                }
                            }
                            Err(e) => break Err(e.into()),
                        }
                    }
                }
            }
        })
        .await;

        match execution_result {
            Ok(Ok((stdout_lines, stderr_lines))) => {
                // Read the output file
                let result = if temp_output.exists() {
                    let output_content = tokio::fs::read_to_string(&temp_output).await?;
                    serde_json::from_str::<PluginResult>(&output_content).unwrap_or_else(|e| {
                        error!("Failed to parse plugin output: {}", e);
                        PluginResult {
                            success: false,
                            output_items: vec![],
                            modified_sequence: None,
                            artifacts: vec![],
                            logs: stderr_lines,
                            error_message: Some(format!("Failed to parse output: {}", e)),
                        }
                    })
                } else {
                    // No output file, create a basic result
                    PluginResult {
                        success: true,
                        output_items: vec![],
                        modified_sequence: None,
                        artifacts: vec![],
                        logs: stdout_lines,
                        error_message: None,
                    }
                };

                // Read additional logs if available
                let mut all_logs = result.logs;
                if temp_logs.exists() {
                    if let Ok(log_content) = tokio::fs::read_to_string(&temp_logs).await {
                        all_logs.extend(log_content.lines().map(|s| s.to_string()));
                    }
                }

                Ok(PluginResult {
                    logs: all_logs,
                    ..result
                })
            }
            Ok(Err(e)) => Err(e),
            Err(_) => Err(PluginError::Timeout(format!(
                "Plugin execution timed out after {:?}",
                self.max_execution_time
            ))
            .into()),
        }
    }

    /// Check if ComfyUI is available and working
    pub async fn check_comfyui_availability(&self) -> bool {
        let check_script = r#"
import sys
try:
    import comfy
    print("ComfyUI available")
    sys.exit(0)
except ImportError:
    print("ComfyUI not available")
    sys.exit(1)
"#;

        match TokioCommand::new(&self.python_executable)
            .arg("-c")
            .arg(check_script)
            .output()
            .await
        {
            Ok(output) => output.status.success(),
            Err(_) => false,
        }
    }

    /// Install Python dependencies for a plugin
    pub async fn install_plugin_dependencies(&self, requirements_file: &Path) -> Result<()> {
        if !requirements_file.exists() {
            return Ok(()); // No requirements file
        }

        debug!(
            "Installing Python dependencies from {:?}",
            requirements_file
        );

        let output = TokioCommand::new(&self.python_executable)
            .arg("-m")
            .arg("pip")
            .arg("install")
            .arg("-r")
            .arg(requirements_file)
            .arg("--user") // Install to user directory to avoid permission issues
            .output()
            .await?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(PluginError::PythonBridge(format!(
                "Failed to install dependencies: {}",
                stderr
            ))
            .into());
        }

        Ok(())
    }

    fn find_python_executable() -> Result<String> {
        // Try different Python executable names
        let candidates = [
            "python3",
            "python",
            "python3.9",
            "python3.10",
            "python3.11",
            "python3.12",
        ];

        for candidate in &candidates {
            if let Ok(output) = std::process::Command::new(candidate)
                .arg("--version")
                .output()
            {
                if output.status.success() {
                    let version_str = String::from_utf8_lossy(&output.stdout);
                    if version_str.contains("Python 3.") {
                        debug!(
                            "Found Python executable: {} ({})",
                            candidate,
                            version_str.trim()
                        );
                        return Ok(candidate.to_string());
                    }
                }
            }
        }

        Err(PluginError::PythonBridge("Python 3.x not found".to_string()).into())
    }

    fn verify_python_environment(python_executable: &str) -> Result<()> {
        // Check basic Python functionality
        let output = std::process::Command::new(python_executable)
            .arg("-c")
            .arg("import json, sys, os; print('Python environment OK')")
            .output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(PluginError::PythonBridge(format!(
                "Python environment verification failed: {}",
                stderr
            ))
            .into());
        }

        debug!("Python environment verified");
        Ok(())
    }
}

/// Helper functions for Python plugin development
pub mod python_helpers {
    use super::*;

    /// Generate a basic Python plugin template
    pub fn generate_python_plugin_template(plugin_name: &str, plugin_type: &str) -> String {
        format!(
            r#"#!/usr/bin/env python3
"""
{plugin_name} - A {plugin_type} plugin for Gausian Native Editor
"""

import json
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List

class {plugin_name}Plugin:
    def __init__(self):
        self.name = "{plugin_name}"
        self.version = "1.0.0"
    
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main plugin processing function.
        
        Args:
            context: Plugin execution context containing sequence, parameters, etc.
            
        Returns:
            Plugin result dictionary
        """
        # Extract context information
        sequence = context.get('sequence', {{}})
        parameters = context.get('parameters', {{}})
        current_frame = context.get('current_frame', 0)
        
        # TODO: Implement your plugin logic here
        print(f"Processing frame {{current_frame}} with parameters: {{parameters}}")
        
        # Return result
        return {{
            "success": True,
            "output_items": [],
            "modified_sequence": None,
            "artifacts": [],
            "logs": [f"{{self.name}} processed successfully"],
            "error_message": None
        }}

def main():
    parser = argparse.ArgumentParser(description='{plugin_name} Plugin')
    parser.add_argument('--input', required=True, help='Input JSON file')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--logs', help='Logs file')
    
    args = parser.parse_args()
    
    # Read input context
    with open(args.input, 'r') as f:
        context = json.load(f)
    
    # Create plugin instance and process
    plugin = {plugin_name}Plugin()
    try:
        result = plugin.process(context)
        
        # Write output
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
            
        # Write logs if specified
        if args.logs:
            with open(args.logs, 'w') as f:
                for log in result.get('logs', []):
                    f.write(f"{{log}}\n")
                    
    except Exception as e:
        # Handle errors
        error_result = {{
            "success": False,
            "output_items": [],
            "modified_sequence": None,
            "artifacts": [],
            "logs": [f"Error: {{str(e)}}"],
            "error_message": str(e)
        }}
        
        with open(args.output, 'w') as f:
            json.dump(error_result, f, indent=2)
        
        sys.exit(1)

if __name__ == "__main__":
    main()
"#,
            plugin_name = plugin_name,
            plugin_type = plugin_type
        )
    }

    /// Generate requirements.txt for a Python plugin
    pub fn generate_requirements_txt(dependencies: &[&str]) -> String {
        let mut requirements = String::new();
        for dep in dependencies {
            requirements.push_str(dep);
            requirements.push('\n');
        }
        requirements
    }
}
