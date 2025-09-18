use crate::{PluginError, PluginManifest};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokio::fs;
use tracing::{debug, error, info};

/// Plugin marketplace client for discovering, downloading, and managing plugins
pub struct PluginMarketplace {
    marketplace_url: String,
    cache_dir: PathBuf,
    client: reqwest::Client,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketplacePlugin {
    pub id: String,
    pub name: String,
    pub version: String,
    pub author: String,
    pub description: String,
    pub category: String,
    pub tags: Vec<String>,
    pub download_url: String,
    pub checksum: String,
    pub manifest: PluginManifest,
    pub screenshots: Vec<String>,
    pub documentation_url: Option<String>,
    pub license: String,
    pub price: Option<f64>, // None for free plugins
    pub rating: Option<f32>,
    pub download_count: u64,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketplaceResponse {
    pub plugins: Vec<MarketplacePlugin>,
    pub total_count: u32,
    pub page: u32,
    pub per_page: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    pub query: Option<String>,
    pub category: Option<String>,
    pub tags: Vec<String>,
    pub free_only: bool,
    pub page: u32,
    pub per_page: u32,
    pub sort_by: SortBy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SortBy {
    Name,
    Rating,
    Downloads,
    Updated,
    Created,
}

impl Default for SearchQuery {
    fn default() -> Self {
        Self {
            query: None,
            category: None,
            tags: vec![],
            free_only: false,
            page: 1,
            per_page: 20,
            sort_by: SortBy::Rating,
        }
    }
}

impl PluginMarketplace {
    pub fn new(marketplace_url: String, cache_dir: PathBuf) -> Result<Self> {
        let client = reqwest::Client::builder()
            .user_agent("GausianNativeEditor/1.0")
            .timeout(std::time::Duration::from_secs(30))
            .build()?;

        Ok(Self {
            marketplace_url,
            cache_dir,
            client,
        })
    }

    /// Search for plugins in the marketplace
    pub async fn search_plugins(&self, query: SearchQuery) -> Result<MarketplaceResponse> {
        let url = format!("{}/api/plugins/search", self.marketplace_url);

        debug!("Searching marketplace with query: {:?}", query);

        let response = self.client.get(&url).json(&query).send().await?;

        if !response.status().is_success() {
            return Err(PluginError::LoadError(format!(
                "Marketplace search failed with status: {}",
                response.status()
            ))
            .into());
        }

        let marketplace_response: MarketplaceResponse = response.json().await?;

        info!("Found {} plugins", marketplace_response.plugins.len());
        Ok(marketplace_response)
    }

    /// Get featured plugins
    pub async fn get_featured_plugins(&self) -> Result<Vec<MarketplacePlugin>> {
        let url = format!("{}/api/plugins/featured", self.marketplace_url);

        let response = self.client.get(&url).send().await?;

        if !response.status().is_success() {
            return Err(PluginError::LoadError(format!(
                "Failed to get featured plugins: {}",
                response.status()
            ))
            .into());
        }

        let plugins: Vec<MarketplacePlugin> = response.json().await?;
        Ok(plugins)
    }

    /// Download and install a plugin from the marketplace
    pub async fn install_plugin(&self, plugin_id: &str, install_dir: &Path) -> Result<PathBuf> {
        info!("Installing plugin: {}", plugin_id);

        // Get plugin details
        let plugin = self.get_plugin_details(plugin_id).await?;

        // Create plugin directory
        let plugin_dir = install_dir.join(&plugin.id);
        fs::create_dir_all(&plugin_dir).await?;

        // Download plugin archive
        let archive_path = self.download_plugin_archive(&plugin).await?;

        // Extract archive
        self.extract_plugin_archive(&archive_path, &plugin_dir)
            .await?;

        // Verify installation
        self.verify_plugin_installation(&plugin_dir, &plugin)
            .await?;

        // Clean up archive
        let _ = fs::remove_file(&archive_path).await;

        info!("Plugin {} installed successfully", plugin_id);
        Ok(plugin_dir)
    }

    /// Get detailed information about a specific plugin
    pub async fn get_plugin_details(&self, plugin_id: &str) -> Result<MarketplacePlugin> {
        let url = format!("{}/api/plugins/{}", self.marketplace_url, plugin_id);

        let response = self.client.get(&url).send().await?;

        if !response.status().is_success() {
            return Err(PluginError::NotFound(format!(
                "Plugin {} not found in marketplace",
                plugin_id
            ))
            .into());
        }

        let plugin: MarketplacePlugin = response.json().await?;
        Ok(plugin)
    }

    /// Check for plugin updates
    pub async fn check_updates(
        &self,
        installed_plugins: &HashMap<String, String>, // plugin_id -> current_version
    ) -> Result<Vec<MarketplacePlugin>> {
        let mut updates = Vec::new();

        for (plugin_id, current_version) in installed_plugins {
            match self.get_plugin_details(plugin_id).await {
                Ok(marketplace_plugin) => {
                    if marketplace_plugin.version != *current_version {
                        debug!(
                            "Update available for {}: {} -> {}",
                            plugin_id, current_version, marketplace_plugin.version
                        );
                        updates.push(marketplace_plugin);
                    }
                }
                Err(e) => {
                    error!("Failed to check updates for {}: {}", plugin_id, e);
                }
            }
        }

        Ok(updates)
    }

    /// Submit a plugin to the marketplace (for plugin developers)
    pub async fn submit_plugin(
        &self,
        plugin_path: &Path,
        metadata: SubmissionMetadata,
        api_key: &str,
    ) -> Result<String> {
        let url = format!("{}/api/plugins/submit", self.marketplace_url);

        // Create multipart form
        let plugin_bytes = tokio::fs::read(plugin_path).await?;
        let plugin_part = reqwest::multipart::Part::bytes(plugin_bytes)
            .file_name("plugin.zip")
            .mime_str("application/zip")?;

        let form = reqwest::multipart::Form::new()
            .text("metadata", serde_json::to_string(&metadata)?)
            .part("plugin_archive", plugin_part);

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .multipart(form)
            .send()
            .await?;

        if !response.status().is_success() {
            let error_text = response.text().await?;
            return Err(PluginError::LoadError(format!(
                "Plugin submission failed: {}",
                error_text
            ))
            .into());
        }

        let submission_response: SubmissionResponse = response.json().await?;
        Ok(submission_response.submission_id)
    }

    async fn download_plugin_archive(&self, plugin: &MarketplacePlugin) -> Result<PathBuf> {
        let cache_path = self
            .cache_dir
            .join(format!("{}-{}.zip", plugin.id, plugin.version));

        // Check if already cached
        if cache_path.exists() {
            debug!("Using cached plugin archive: {:?}", cache_path);
            return Ok(cache_path);
        }

        // Ensure cache directory exists
        fs::create_dir_all(&self.cache_dir).await?;

        debug!("Downloading plugin from: {}", plugin.download_url);

        let response = self.client.get(&plugin.download_url).send().await?;

        if !response.status().is_success() {
            return Err(PluginError::LoadError(format!(
                "Failed to download plugin: {}",
                response.status()
            ))
            .into());
        }

        let bytes = response.bytes().await?;

        // Verify checksum
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let actual_checksum = format!("{:x}", hasher.finalize());

        if actual_checksum != plugin.checksum {
            return Err(PluginError::SecurityViolation(format!(
                "Plugin checksum mismatch. Expected: {}, Got: {}",
                plugin.checksum, actual_checksum
            ))
            .into());
        }

        fs::write(&cache_path, &bytes).await?;
        Ok(cache_path)
    }

    async fn extract_plugin_archive(&self, archive_path: &Path, target_dir: &Path) -> Result<()> {
        use std::io::Read;

        // For now, assume ZIP archives. In a real implementation, you'd detect the format
        let file = std::fs::File::open(archive_path)?;
        let mut archive = zip::ZipArchive::new(file)?;

        for i in 0..archive.len() {
            let mut file = archive.by_index(i)?;
            let outpath = target_dir.join(file.name());

            if file.name().ends_with('/') {
                fs::create_dir_all(&outpath).await?;
            } else {
                if let Some(p) = outpath.parent() {
                    fs::create_dir_all(p).await?;
                }

                // Read file contents synchronously then write asynchronously
                let mut contents = Vec::new();
                file.read_to_end(&mut contents)?;
                fs::write(&outpath, contents).await?;
            }
        }

        Ok(())
    }

    async fn verify_plugin_installation(
        &self,
        plugin_dir: &Path,
        expected_plugin: &MarketplacePlugin,
    ) -> Result<()> {
        // Check if manifest exists
        let manifest_path = plugin_dir.join("plugin.json");
        if !manifest_path.exists() {
            return Err(PluginError::InvalidManifest(
                "plugin.json not found after installation".to_string(),
            )
            .into());
        }

        // Verify manifest matches expected plugin
        let manifest_content = fs::read_to_string(&manifest_path).await?;
        let manifest: PluginManifest = serde_json::from_str(&manifest_content)?;

        if manifest.name != expected_plugin.manifest.name
            || manifest.version != expected_plugin.manifest.version
        {
            return Err(PluginError::InvalidManifest(
                "Installed plugin manifest doesn't match expected plugin".to_string(),
            )
            .into());
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmissionMetadata {
    pub name: String,
    pub description: String,
    pub category: String,
    pub tags: Vec<String>,
    pub license: String,
    pub documentation_url: Option<String>,
    pub screenshots: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubmissionResponse {
    pub submission_id: String,
    pub status: String,
    pub message: String,
}

/// Mock marketplace for development and testing
pub struct MockMarketplace {
    plugins: Vec<MarketplacePlugin>,
}

impl MockMarketplace {
    pub fn new() -> Self {
        Self {
            plugins: Self::create_sample_plugins(),
        }
    }

    pub async fn search_plugins(&self, query: SearchQuery) -> Result<MarketplaceResponse> {
        let mut filtered_plugins = self.plugins.clone();

        // Apply filters
        if let Some(ref q) = query.query {
            let q_lower = q.to_lowercase();
            filtered_plugins.retain(|p| {
                p.name.to_lowercase().contains(&q_lower)
                    || p.description.to_lowercase().contains(&q_lower)
                    || p.tags.iter().any(|t| t.to_lowercase().contains(&q_lower))
            });
        }

        if let Some(ref category) = query.category {
            filtered_plugins.retain(|p| p.category == *category);
        }

        if query.free_only {
            filtered_plugins.retain(|p| p.price.is_none());
        }

        // Apply sorting
        match query.sort_by {
            SortBy::Name => filtered_plugins.sort_by(|a, b| a.name.cmp(&b.name)),
            SortBy::Rating => filtered_plugins.sort_by(|a, b| {
                b.rating
                    .unwrap_or(0.0)
                    .partial_cmp(&a.rating.unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            }),
            SortBy::Downloads => {
                filtered_plugins.sort_by(|a, b| b.download_count.cmp(&a.download_count))
            }
            SortBy::Updated => filtered_plugins.sort_by(|a, b| b.updated_at.cmp(&a.updated_at)),
            SortBy::Created => filtered_plugins.sort_by(|a, b| b.created_at.cmp(&a.created_at)),
        }

        // Apply pagination
        let start = ((query.page - 1) * query.per_page) as usize;
        let end = (start + query.per_page as usize).min(filtered_plugins.len());
        let page_plugins = filtered_plugins[start..end].to_vec();

        Ok(MarketplaceResponse {
            plugins: page_plugins,
            total_count: filtered_plugins.len() as u32,
            page: query.page,
            per_page: query.per_page,
        })
    }

    fn create_sample_plugins() -> Vec<MarketplacePlugin> {
        vec![
            MarketplacePlugin {
                id: "blur-effect".to_string(),
                name: "Gaussian Blur".to_string(),
                version: "1.0.0".to_string(),
                author: "Gausian Team".to_string(),
                description: "High-quality Gaussian blur effect with adjustable radius".to_string(),
                category: "Effects".to_string(),
                tags: vec!["blur".to_string(), "effect".to_string()],
                download_url: "https://example.com/plugins/blur-effect.zip".to_string(),
                checksum: "abc123".to_string(),
                manifest: crate::utils::create_plugin_manifest(
                    "Gaussian Blur",
                    "1.0.0",
                    "Gausian Team",
                    crate::PluginType::Effect,
                    crate::PluginRuntime::Wasm,
                    "blur.wasm",
                ),
                screenshots: vec!["https://example.com/screenshots/blur1.png".to_string()],
                documentation_url: Some("https://docs.example.com/blur-effect".to_string()),
                license: "MIT".to_string(),
                price: None,
                rating: Some(4.5),
                download_count: 1250,
                created_at: "2024-01-15T10:00:00Z".to_string(),
                updated_at: "2024-03-01T15:30:00Z".to_string(),
            },
            // Add more sample plugins...
        ]
    }
}
