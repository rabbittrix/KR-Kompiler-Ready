use crate::utils::fs_utils;
use regex;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;
use std::{thread, time::Duration};

#[derive(Debug, Serialize)]
struct OllamaRequest {
    model: String,
    prompt: String,
    stream: bool,
}

#[derive(Debug, Deserialize)]
struct OllamaResponse {
    _model: String,
    _created_at: String,
    response: String,
    _done: bool,
}

pub struct OllamaAI {
    base_url: String,
}

impl OllamaAI {
    pub fn new() -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
        }
    }

    /// Check if Ollama is running and accessible
    pub fn is_available(&self) -> bool {
        let client = reqwest::blocking::Client::new();
        let response = client.get(&format!("{}/api/tags", self.base_url)).send();
        response.is_ok()
    }

    /// Start/pull a model to ensure it's available and running
    pub fn start_model(&self, model: &str) -> Result<(), String> {
        println!("🤖 Ensuring model '{}' is running...", model);

        // First, try to pull the model if it doesn't exist
        println!("📥 Checking if model '{}' is available...", model);
        let pull_output = Command::new("ollama").args(&["pull", model]).output();

        match pull_output {
            Ok(output) => {
                if output.status.success() {
                    println!("✅ Model '{}' is available!", model);
                } else {
                    println!("⚠️ Model '{}' pull had issues, but continuing...", model);
                }
            }
            Err(_) => {
                println!("⚠️ Could not pull model '{}', but continuing...", model);
            }
        }

        // Try to check if the model is available via the API
        let client = reqwest::blocking::Client::new();
        let tags_url = format!("{}/api/tags", self.base_url);
        let mut model_running = false;

        // Check if model is already running
        if let Ok(response) = client.get(&tags_url).send() {
            if let Ok(text) = response.text() {
                if text.contains(model) {
                    model_running = true;
                    println!("✅ Model '{}' is already running!", model);
                }
            }
        }

        if !model_running {
            // Start the model in the background
            println!(
                "🔄 Starting model '{}' with 'ollama run {}' in the background...",
                model, model
            );
            #[cfg(target_os = "windows")]
            {
                // On Windows, use 'start /B' to run in background
                let _ = std::process::Command::new("cmd")
                    .args(["/C", &format!("start /B ollama run {}", model)])
                    .spawn();
            }
            #[cfg(not(target_os = "windows"))]
            {
                // On Unix, use '&' to run in background
                let _ = std::process::Command::new("sh")
                    .arg("-c")
                    .arg(format!("ollama run {} &", model))
                    .spawn();
            }

            // Wait for the model to be available
            println!("⏳ Waiting for model '{}' to be ready...", model);
            let mut waited = 0;
            let max_wait = 60; // seconds (increased timeout)
            loop {
                thread::sleep(Duration::from_secs(2));
                waited += 2;

                // Try to make a simple request to test if model is ready
                let test_request = OllamaRequest {
                    model: model.to_string(),
                    prompt: "Hello".to_string(),
                    stream: false,
                };

                if let Ok(request_json) = serde_json::to_string(&test_request) {
                    if let Ok(response) = client
                        .post(&format!("{}/api/generate", self.base_url))
                        .header("Content-Type", "application/json")
                        .body(request_json)
                        .send()
                    {
                        if response.status().is_success() {
                            println!("✅ Model '{}' is now ready and responding!", model);
                            break;
                        }
                    }
                }

                if waited >= max_wait {
                    return Err(format!(
                        "Timed out waiting for model '{}' to start. Please check if Ollama is running and the model exists.",
                        model
                    ));
                }

                if waited % 10 == 0 {
                    println!(
                        "⏳ Still waiting for model '{}'... ({}s elapsed)",
                        model, waited
                    );
                }
            }
        }

        Ok(())
    }

    /// Get list of available models
    pub fn list_models(&self) -> Result<Vec<String>, String> {
        // First try using ollama CLI
        let output = Command::new("ollama").arg("list").output();

        match output {
            Ok(output) if output.status.success() => {
                let output_str = String::from_utf8_lossy(&output.stdout);
                let models = self.parse_ollama_list(&output_str);
                Ok(models)
            }
            _ => {
                // Fallback to API call
                let client = reqwest::blocking::Client::new();
                let response = client
                    .get(&format!("{}/api/tags", self.base_url))
                    .send()
                    .map_err(|e| format!("Failed to connect to Ollama: {}", e))?;

                if response.status().is_success() {
                    let response_text = response
                        .text()
                        .map_err(|e| format!("Failed to read response: {}", e))?;

                    let models: serde_json::Value = serde_json::from_str(&response_text)
                        .map_err(|e| format!("Failed to parse response: {}", e))?;

                    if let Some(models_array) = models.get("models").and_then(|m| m.as_array()) {
                        let model_names: Vec<String> = models_array
                            .iter()
                            .filter_map(|model| {
                                model
                                    .get("name")
                                    .and_then(|n| n.as_str())
                                    .map(|s| s.to_string())
                            })
                            .collect();
                        Ok(model_names)
                    } else {
                        Err("Invalid response format from Ollama API".to_string())
                    }
                } else {
                    Err("Failed to get models from Ollama API".to_string())
                }
            }
        }
    }

    /// Parse ollama list output
    fn parse_ollama_list(&self, output: &str) -> Vec<String> {
        let lines: Vec<&str> = output.lines().collect();
        let mut models = Vec::new();

        // Skip header line
        for line in lines.iter().skip(1) {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                models.push(parts[0].to_string());
            }
        }

        models
    }

    /// Generate project structure using AI
    pub fn generate_project(&self, model: &str, prompt: &str) -> Result<String, String> {
        let client = reqwest::blocking::Client::new();

        let request = OllamaRequest {
            model: model.to_string(),
            prompt: self.build_project_prompt(prompt),
            stream: false,
        };

        let request_json = serde_json::to_string(&request)
            .map_err(|e| format!("Failed to serialize request: {}", e))?;

        let response = client
            .post(&format!("{}/api/generate", self.base_url))
            .header("Content-Type", "application/json")
            .body(request_json)
            .send()
            .map_err(|e| format!("Failed to send request to Ollama: {}", e))?;

        if response.status().is_success() {
            let response_text = response
                .text()
                .map_err(|e| format!("Failed to read response: {}", e))?;

            let ollama_response: OllamaResponse = serde_json::from_str(&response_text)
                .map_err(|e| format!("Failed to parse response: {}", e))?;

            Ok(ollama_response.response)
        } else {
            Err(format!("Ollama API returned error: {}", response.status()))
        }
    }

    /// Generate actual project files using AI
    pub fn generate_project_files(
        &self,
        model: &str,
        project_name: &str,
        description: &str,
        path: &Path,
    ) -> Result<ProjectConfig, String> {
        // Start the model first
        self.start_model(model)?;

        let client = reqwest::blocking::Client::new();

        let request = OllamaRequest {
            model: model.to_string(),
            prompt: self.build_file_generation_prompt(project_name, description),
            stream: false,
        };

        let request_json = serde_json::to_string(&request)
            .map_err(|e| format!("Failed to serialize request: {}", e))?;

        let response = client
            .post(&format!("{}/api/generate", self.base_url))
            .header("Content-Type", "application/json")
            .body(request_json)
            .send()
            .map_err(|e| format!("Failed to send request to Ollama: {}", e))?;

        if response.status().is_success() {
            let response_text = response
                .text()
                .map_err(|e| format!("Failed to read response: {}", e))?;

            let ollama_response: OllamaResponse = serde_json::from_str(&response_text)
                .map_err(|e| format!("Failed to parse response: {}", e))?;

            // Parse the AI response and create files
            self.parse_and_create_files(&ollama_response.response, path)
        } else {
            Err(format!("Ollama API returned error: {}", response.status()))
        }
    }

    /// Interactive AI session for project creation
    pub fn interactive_project_creation(
        &self,
        model: &str,
        project_name: &str,
        path: &Path,
    ) -> Result<ProjectConfig, String> {
        // Start the model first
        self.start_model(model)?;

        println!(
            "🤖 Starting interactive AI session for project '{}'",
            project_name
        );
        println!(
            "💬 You can ask questions, request changes, or ask for clarification at any time."
        );
        println!("💬 Type 'done' when you're satisfied with the project, or 'cancel' to abort.\n");

        let mut current_config = ProjectConfig::default();
        current_config.description = format!("AI-generated project: {}", project_name);

        loop {
            let user_input: String = dialoguer::Input::new()
                .with_prompt("What would you like to do? (describe features, ask questions, or type 'done'/'cancel')")
                .interact_text()
                .map_err(|_| "Failed to get user input".to_string())?;

            match user_input.trim().to_lowercase().as_str() {
                "done" => {
                    println!("✅ Finalizing project creation...");
                    break;
                }
                "cancel" => {
                    return Err("Project creation canceled by user".to_string());
                }
                _ => {
                    // Generate response based on user input
                    let ai_response =
                        self.generate_interactive_response(model, &user_input, &current_config)?;
                    println!("🤖 AI: {}", ai_response);

                    // Ask if user wants to apply AI suggestions
                    if dialoguer::Confirm::new()
                        .with_prompt("Apply these AI suggestions?")
                        .default(true)
                        .interact()
                        .map_err(|_| "Failed to get user confirmation".to_string())?
                    {
                        current_config =
                            self.update_config_from_response(&ai_response, &current_config);
                        println!("✅ Configuration updated!");
                    }
                }
            }
        }

        // Generate final project files
        self.generate_final_project_files(model, &current_config, path)?;

        Ok(current_config)
    }

    /// Generate interactive response from AI
    fn generate_interactive_response(
        &self,
        model: &str,
        user_input: &str,
        current_config: &ProjectConfig,
    ) -> Result<String, String> {
        let client = reqwest::blocking::Client::new();

        let prompt = format!(
            r#"You are an expert Python developer helping to create a project. 

Current project configuration:
- Python version: {}
- Project type: {}
- Dependencies: {}

User request: "{}"

Please provide helpful guidance, suggestions, or answer questions. Be specific and actionable. If suggesting changes, explain why they would be beneficial."#,
            current_config.python_version,
            current_config.project_type,
            current_config.dependencies.join(", "),
            user_input
        );

        let request = OllamaRequest {
            model: model.to_string(),
            prompt,
            stream: false,
        };

        let request_json = serde_json::to_string(&request)
            .map_err(|e| format!("Failed to serialize request: {}", e))?;

        let response = client
            .post(&format!("{}/api/generate", self.base_url))
            .header("Content-Type", "application/json")
            .body(request_json)
            .send()
            .map_err(|e| format!("Failed to send request to Ollama: {}", e))?;

        if response.status().is_success() {
            let response_text = response
                .text()
                .map_err(|e| format!("Failed to read response: {}", e))?;

            let ollama_response: OllamaResponse = serde_json::from_str(&response_text)
                .map_err(|e| format!("Failed to parse response: {}", e))?;

            Ok(ollama_response.response)
        } else {
            Err(format!("Ollama API returned error: {}", response.status()))
        }
    }

    /// Update configuration based on AI response
    fn update_config_from_response(
        &self,
        response: &str,
        current_config: &ProjectConfig,
    ) -> ProjectConfig {
        let mut new_config = current_config.clone();

        // Extract Python version
        if response.contains("Python 3.12") {
            new_config.python_version = "3.12".to_string();
        } else if response.contains("Python 3.11") {
            new_config.python_version = "3.11".to_string();
        } else if response.contains("Python 3.10") {
            new_config.python_version = "3.10".to_string();
        }

        // Extract project type
        if response.to_lowercase().contains("api") {
            new_config.project_type = "API".to_string();
        } else if response.to_lowercase().contains("web") {
            new_config.project_type = "Web App".to_string();
        } else if response.to_lowercase().contains("cli") {
            new_config.project_type = "CLI".to_string();
        } else if response.to_lowercase().contains("data") || response.to_lowercase().contains("ml")
        {
            new_config.project_type = "Data Science".to_string();
        }

        // Extract new dependencies
        let new_deps = self.extract_dependencies_advanced(response);
        for dep in new_deps {
            if !new_config.dependencies.contains(&dep) {
                new_config.dependencies.push(dep);
            }
        }

        new_config
    }

    /// Generate final project files
    fn generate_final_project_files(
        &self,
        model: &str,
        config: &ProjectConfig,
        path: &Path,
    ) -> Result<(), String> {
        println!("🤖 Generating project files...");

        // Generate main application file
        let main_content = self.generate_main_file(model, config)?;
        fs_utils::write_file(&path.join("main.py"), main_content)
            .map_err(|_| "Failed to write main.py".to_string())?;

        // Generate requirements.txt
        let requirements_content = config.dependencies.join("\n");
        fs_utils::write_file(&path.join("requirements.txt"), requirements_content)
            .map_err(|_| "Failed to write requirements.txt".to_string())?;

        // Generate README.md
        let readme_content = self.generate_readme(model, config)?;
        fs_utils::write_file(&path.join("README.md"), readme_content)
            .map_err(|_| "Failed to write README.md".to_string())?;

        println!("✅ Project files generated successfully!");
        Ok(())
    }

    /// Generate main application file
    fn generate_main_file(&self, model: &str, config: &ProjectConfig) -> Result<String, String> {
        let client = reqwest::blocking::Client::new();

        let prompt = format!(
            r#"Create a complete, working Python main.py file for a {} project.

Requirements:
- Python version: {}
- Dependencies: {}
- Project type: {}

Generate a complete, production-ready main.py file with proper imports, error handling, and documentation. The code should be ready to run immediately."#,
            config.project_type,
            config.python_version,
            config.dependencies.join(", "),
            config.project_type
        );

        let request = OllamaRequest {
            model: model.to_string(),
            prompt,
            stream: false,
        };

        let request_json = serde_json::to_string(&request)
            .map_err(|e| format!("Failed to serialize request: {}", e))?;

        let response = client
            .post(&format!("{}/api/generate", self.base_url))
            .header("Content-Type", "application/json")
            .body(request_json)
            .send()
            .map_err(|e| format!("Failed to send request to Ollama: {}", e))?;

        if response.status().is_success() {
            let response_text = response
                .text()
                .map_err(|e| format!("Failed to read response: {}", e))?;

            let ollama_response: OllamaResponse = serde_json::from_str(&response_text)
                .map_err(|e| format!("Failed to parse response: {}", e))?;

            Ok(ollama_response.response)
        } else {
            Err(format!("Ollama API returned error: {}", response.status()))
        }
    }

    /// Generate README file
    fn generate_readme(&self, model: &str, config: &ProjectConfig) -> Result<String, String> {
        let client = reqwest::blocking::Client::new();

        let prompt = format!(
            r#"Create a comprehensive README.md file for a {} Python project.

Project details:
- Python version: {}
- Dependencies: {}
- Project type: {}

Include:
1. Project description
2. Installation instructions
3. Usage examples
4. Dependencies list
5. How to run the project
6. Any additional setup steps"#,
            config.project_type,
            config.python_version,
            config.dependencies.join(", "),
            config.project_type
        );

        let request = OllamaRequest {
            model: model.to_string(),
            prompt,
            stream: false,
        };

        let request_json = serde_json::to_string(&request)
            .map_err(|e| format!("Failed to serialize request: {}", e))?;

        let response = client
            .post(&format!("{}/api/generate", self.base_url))
            .header("Content-Type", "application/json")
            .body(request_json)
            .send()
            .map_err(|e| format!("Failed to send request to Ollama: {}", e))?;

        if response.status().is_success() {
            let response_text = response
                .text()
                .map_err(|e| format!("Failed to read response: {}", e))?;

            let ollama_response: OllamaResponse = serde_json::from_str(&response_text)
                .map_err(|e| format!("Failed to parse response: {}", e))?;

            Ok(ollama_response.response)
        } else {
            Err(format!("Ollama API returned error: {}", response.status()))
        }
    }

    /// Parse AI response and create files
    fn parse_and_create_files(&self, response: &str, path: &Path) -> Result<ProjectConfig, String> {
        println!("🤖 Parsing AI response and creating files...");

        // Extract configuration from response
        let config = self.parse_ai_response(response);

        // Extract file content from the AI response
        let files = self.extract_files_from_response(response);

        // Check if files are empty before the loop
        let files_empty = files.is_empty();

        // Create each file
        for (filename, content) in files {
            let file_path = path.join(&filename);

            // Ensure directory exists
            if let Some(parent) = file_path.parent() {
                if !parent.exists() {
                    std::fs::create_dir_all(parent)
                        .map_err(|e| format!("Failed to create directory: {}", e))?;
                }
            }

            // Write file content
            fs_utils::write_file(&file_path, content)
                .map_err(|e| format!("Failed to write {}: {}", filename, e))?;

            println!("✅ Created: {}", filename);
        }

        // Ensure app.py or main.py exists
        let app_py = path.join("app.py");
        let main_py = path.join("main.py");
        if !app_py.exists() && !main_py.exists() {
            self.create_basic_structure(path, &config)?;
        }

        // Ensure requirements.txt exists
        let reqs = path.join("requirements.txt");
        if !reqs.exists() {
            let requirements = if config.dependencies.is_empty() {
                "fastapi\nuvicorn\npydantic".to_string()
            } else {
                config.dependencies.join("\n")
            };
            fs_utils::write_file(&reqs, requirements)
                .map_err(|e| format!("Failed to write requirements.txt: {}", e))?;
        }

        // If no files were extracted, create basic structure
        if files_empty {
            println!("⚠️ No files extracted from AI response, creating basic structure...");
            self.create_basic_structure(path, &config)?;
        }

        Ok(config)
    }

    /// Build a comprehensive prompt for project generation
    fn build_project_prompt(&self, user_prompt: &str) -> String {
        format!(
            r#"You are an expert Python developer and project architect. A user wants to create a Python project with the following requirements:

"{user_prompt}"

IMPORTANT: Create EXACTLY what the user requests. If they ask for 50 items, provide 50 items. If they ask for a database, include database code. If they ask for specific features, implement them precisely.

Please provide a detailed project structure and implementation plan in the following JSON format:

{{
    "project_type": "API|Web App|CLI|Data Science|ML|DL|Microservices|General",
    "python_version": "3.9|3.10|3.11|3.12|3.13",
    "dependencies": [
        {{
            "name": "package_name",
            "version": "version_specifier",
            "purpose": "brief description"
        }}
    ],
    "file_structure": [
        {{
            "path": "file_path",
            "type": "file|directory",
            "description": "purpose of this file/directory"
        }}
    ],
    "main_features": [
        "feature 1",
        "feature 2"
    ],
    "configuration": {{
        "database": "sqlite|postgresql|mysql|none",
        "authentication": "jwt|session|none",
        "api_framework": "fastapi|flask|django|none",
        "frontend": "streamlit|dash|html|none"
    }},
    "deployment": {{
        "platform": "local|docker|cloud",
        "requirements": ["requirement1", "requirement2"]
    }}
}}

Be specific and production-ready. Focus on creating exactly what the user requested, not generic templates."#
        )
    }

    /// Build prompt for file generation
    fn build_file_generation_prompt(&self, project_name: &str, description: &str) -> String {
        format!(
            r#"Create a complete Python project structure for: {}

Description: {}

CRITICAL REQUIREMENTS:
- Create EXACTLY what the user requested
- If they ask for specific numbers (like 50 items), provide exactly that many
- If they ask for specific features, implement them precisely
- If they mention a database, include proper database code
- Use localhost (not 0.0.0.0) for all server configurations
- Include ALL necessary dependencies in requirements.txt
- Make the code production-ready with proper error handling

Please provide the following files in a structured format:

## main.py or app.py
[Complete main application file with all necessary imports, error handling, and functionality. Use localhost for server configuration.]

## requirements.txt
[ALL dependencies with versions, one per line. Include every package needed for the project.]

## README.md
[Comprehensive README with installation, usage, and deployment instructions]

## Additional files as needed
[Any other necessary files for the project - models, schemas, database config, etc.]

Make sure all code is production-ready, follows Python best practices, includes proper error handling, logging, and documentation. Create exactly what the user requested, not generic templates."#,
            project_name, description
        )
    }

    /// Enhanced AI response parser with JSON support
    pub fn parse_ai_response(&self, response: &str) -> ProjectConfig {
        let mut config = ProjectConfig::default();

        // Try to parse JSON response first
        if let Ok(json_config) = self.parse_json_response(response) {
            return json_config;
        }

        // Fallback to text-based parsing
        self.parse_text_response(response, &mut config);
        config
    }

    /// Parse JSON-structured AI response
    fn parse_json_response(&self, response: &str) -> Result<ProjectConfig, serde_json::Error> {
        // Extract JSON from response (handle cases where JSON is embedded in text)
        let json_start = response.find('{');
        let json_end = response.rfind('}');

        if let (Some(start), Some(end)) = (json_start, json_end) {
            let json_str = &response[start..=end];

            #[derive(serde::Deserialize)]
            struct AIProjectPlan {
                project_type: Option<String>,
                python_version: Option<String>,
                dependencies: Option<Vec<DependencyInfo>>,
                _file_structure: Option<Vec<FileInfo>>,
                main_features: Option<Vec<String>>,
                configuration: Option<ConfigInfo>,
                deployment: Option<DeploymentInfo>,
            }

            #[derive(serde::Deserialize)]
            struct DependencyInfo {
                name: String,
                version: Option<String>,
                _purpose: Option<String>,
            }

            #[derive(serde::Deserialize)]
            struct FileInfo {
                _path: String,
                #[serde(rename = "type")]
                _file_type: Option<String>,
                _description: Option<String>,
            }

            #[derive(serde::Deserialize)]
            struct ConfigInfo {
                database: Option<String>,
                authentication: Option<String>,
                api_framework: Option<String>,
                frontend: Option<String>,
            }

            #[derive(serde::Deserialize)]
            struct DeploymentInfo {
                platform: Option<String>,
                _requirements: Option<Vec<String>>,
            }

            let plan: AIProjectPlan = serde_json::from_str(json_str)?;

            let mut config = ProjectConfig::default();

            // Set project type
            if let Some(proj_type) = plan.project_type {
                config.project_type = proj_type;
            }

            // Set Python version
            if let Some(py_version) = plan.python_version {
                config.python_version = py_version;
            }

            // Extract dependencies
            if let Some(deps) = plan.dependencies {
                config.dependencies = deps
                    .into_iter()
                    .map(|d| {
                        if let Some(version) = d.version {
                            format!("{}=={}", d.name, version)
                        } else {
                            d.name
                        }
                    })
                    .collect();
            }

            // Build description from features and configuration
            let mut description_parts = Vec::new();

            if let Some(features) = plan.main_features {
                description_parts.push(format!("Features: {}", features.join(", ")));
            }

            if let Some(config_info) = plan.configuration {
                let mut config_parts = Vec::new();
                if let Some(db) = config_info.database {
                    config_parts.push(format!("Database: {}", db));
                }
                if let Some(auth) = config_info.authentication {
                    config_parts.push(format!("Auth: {}", auth));
                }
                if let Some(api) = config_info.api_framework {
                    config_parts.push(format!("API: {}", api));
                }
                if let Some(frontend) = config_info.frontend {
                    config_parts.push(format!("Frontend: {}", frontend));
                }
                if !config_parts.is_empty() {
                    description_parts.push(format!("Configuration: {}", config_parts.join(", ")));
                }
            }

            if let Some(deployment) = plan.deployment {
                if let Some(platform) = deployment.platform {
                    description_parts.push(format!("Deployment: {}", platform));
                }
            }

            config.description = description_parts.join(". ");

            Ok(config)
        } else {
            Err(serde_json::from_str::<serde_json::Value>("{}").unwrap_err())
        }
    }

    /// Enhanced text-based response parsing
    fn parse_text_response(&self, response: &str, config: &mut ProjectConfig) {
        // Extract Python version with more sophisticated pattern matching
        let version_patterns = [
            (r"Python\s+3\.12", "3.12"),
            (r"Python\s+3\.11", "3.11"),
            (r"Python\s+3\.10", "3.10"),
            (r"Python\s+3\.9", "3.9"),
            (r"python\s*3\.12", "3.12"),
            (r"python\s*3\.11", "3.11"),
            (r"python\s*3\.10", "3.10"),
            (r"python\s*3\.9", "3.9"),
        ];

        for (pattern, version) in version_patterns {
            if regex::Regex::new(pattern).unwrap().is_match(response) {
                config.python_version = version.to_string();
                break;
            }
        }

        // Enhanced project type detection
        let project_patterns = [
            (r"\bAPI\b", "API"),
            (r"\bWeb\s+App\b", "Web App"),
            (r"\bCLI\b", "CLI"),
            (r"\bData\s+Science\b", "Data Science"),
            (r"\bMachine\s+Learning\b", "ML"),
            (r"\bDeep\s+Learning\b", "DL"),
            (r"\bMicroservices\b", "Microservices"),
            (r"\bDjango\b", "Django"),
            (r"\bStreamlit\b", "Streamlit"),
            (r"\bFastAPI\b", "FastAPI"),
        ];

        for (pattern, proj_type) in project_patterns {
            if regex::Regex::new(pattern).unwrap().is_match(response) {
                config.project_type = proj_type.to_string();
                break;
            }
        }

        // Enhanced dependency extraction with version detection
        config.dependencies = self.extract_dependencies_advanced(response);

        // Extract description from response
        config.description = self.extract_description(response);
    }

    /// Advanced dependency extraction with version detection
    fn extract_dependencies_advanced(&self, response: &str) -> Vec<String> {
        let mut deps = Vec::new();

        // Common Python packages with version patterns
        let package_patterns = [
            (r"flask[=<>]\s*[\d\.]+", "flask"),
            (r"fastapi[=<>]\s*[\d\.]+", "fastapi"),
            (r"django[=<>]\s*[\d\.]+", "django"),
            (r"requests[=<>]\s*[\d\.]+", "requests"),
            (r"pandas[=<>]\s*[\d\.]+", "pandas"),
            (r"numpy[=<>]\s*[\d\.]+", "numpy"),
            (r"scikit-learn[=<>]\s*[\d\.]+", "scikit-learn"),
            (r"matplotlib[=<>]\s*[\d\.]+", "matplotlib"),
            (r"seaborn[=<>]\s*[\d\.]+", "seaborn"),
            (r"pytest[=<>]\s*[\d\.]+", "pytest"),
            (r"black[=<>]\s*[\d\.]+", "black"),
            (r"ruff[=<>]\s*[\d\.]+", "ruff"),
            (r"mypy[=<>]\s*[\d\.]+", "mypy"),
            (r"sqlalchemy[=<>]\s*[\d\.]+", "sqlalchemy"),
            (r"alembic[=<>]\s*[\d\.]+", "alembic"),
            (r"uvicorn[=<>]\s*[\d\.]+", "uvicorn"),
            (r"click[=<>]\s*[\d\.]+", "click"),
            (r"rich[=<>]\s*[\d\.]+", "rich"),
            (r"typer[=<>]\s*[\d\.]+", "typer"),
            (r"pydantic[=<>]\s*[\d\.]+", "pydantic"),
            (r"jinja2[=<>]\s*[\d\.]+", "jinja2"),
            (r"aiofiles[=<>]\s*[\d\.]+", "aiofiles"),
            (r"httpx[=<>]\s*[\d\.]+", "httpx"),
            (r"streamlit[=<>]\s*[\d\.]+", "streamlit"),
            (r"torch[=<>]\s*[\d\.]+", "torch"),
            (r"tensorflow[=<>]\s*[\d\.]+", "tensorflow"),
            (r"transformers[=<>]\s*[\d\.]+", "transformers"),
        ];

        // Extract dependencies with versions
        for (pattern, _package) in package_patterns {
            if let Ok(regex) = regex::Regex::new(pattern) {
                if let Some(cap) = regex.captures(response) {
                    deps.push(cap[0].to_string());
                }
            }
        }

        // Also check for simple mentions without versions
        let simple_packages = [
            "flask",
            "fastapi",
            "django",
            "requests",
            "pandas",
            "numpy",
            "scikit-learn",
            "matplotlib",
            "seaborn",
            "pytest",
            "black",
            "ruff",
            "mypy",
            "sqlalchemy",
            "alembic",
            "uvicorn",
            "click",
            "rich",
            "typer",
            "pydantic",
            "jinja2",
            "aiofiles",
            "httpx",
            "streamlit",
            "torch",
            "tensorflow",
            "transformers",
        ];

        for package in &simple_packages {
            if response.to_lowercase().contains(package)
                && !deps.iter().any(|d| d.starts_with(package))
            {
                deps.push(package.to_string());
            }
        }

        deps
    }

    /// Extract meaningful description from AI response
    fn extract_description(&self, response: &str) -> String {
        // Try to extract the first meaningful paragraph
        let lines: Vec<&str> = response.lines().collect();
        let mut description_lines = Vec::new();

        for line in lines {
            let trimmed = line.trim();
            if !trimmed.is_empty()
                && !trimmed.starts_with('#')
                && !trimmed.starts_with('{')
                && !trimmed.starts_with('}')
                && !trimmed.starts_with('"')
                && trimmed.len() > 20
            {
                description_lines.push(trimmed);
                if description_lines.len() >= 3 {
                    break;
                }
            }
        }

        if description_lines.is_empty() {
            "AI-generated Python project".to_string()
        } else {
            description_lines.join(" ")
        }
    }

    /// Enhanced error handling for AI interactions
    pub fn handle_ai_error(&self, error: &str) -> String {
        match error {
            e if e.contains("connection") || e.contains("timeout") => {
                "❌ Connection to AI service failed. Please check if Ollama is running and try again.".to_string()
            }
            e if e.contains("model") || e.contains("not found") => {
                "❌ AI model not found. Please install the model with 'ollama pull <model_name>'.".to_string()
            }
            e if e.contains("parse") || e.contains("json") => {
                "⚠️ AI response could not be parsed. Using fallback configuration.".to_string()
            }
            e if e.contains("rate limit") || e.contains("quota") => {
                "⚠️ AI service rate limit reached. Please wait a moment and try again.".to_string()
            }
            _ => {
                format!("❌ AI error: {}. Please try again or use manual configuration.", error)
            }
        }
    }

    /// Validate AI-generated configuration
    pub fn validate_config(&self, config: &ProjectConfig) -> Result<(), String> {
        // Validate Python version
        let valid_versions = ["3.9", "3.10", "3.11", "3.12", "3.13"];
        if !valid_versions.contains(&config.python_version.as_str()) {
            return Err(format!("Invalid Python version: {}", config.python_version));
        }

        // Validate project type
        let valid_types = [
            "API",
            "Web App",
            "CLI",
            "Data Science",
            "ML",
            "DL",
            "Microservices",
            "Django",
            "Streamlit",
            "FastAPI",
            "General",
        ];
        if !valid_types.contains(&config.project_type.as_str()) {
            return Err(format!("Invalid project type: {}", config.project_type));
        }

        // Validate dependencies (basic check)
        for dep in &config.dependencies {
            if dep.trim().is_empty() {
                return Err("Empty dependency found".to_string());
            }
        }

        Ok(())
    }

    /// Extract file content from AI response
    fn extract_files_from_response(&self, response: &str) -> Vec<(String, String)> {
        let mut files = Vec::new();
        let lines: Vec<&str> = response.lines().collect();
        let mut current_file: Option<String> = None;
        let mut current_content = Vec::new();

        for line in lines {
            if line.starts_with("## ") {
                // Save previous file if exists
                if let Some(filename) = &current_file {
                    let content = current_content.join("\n");
                    if !content.trim().is_empty() {
                        files.push((filename.clone(), content));
                    }
                }

                // Start new file
                let filename = line[3..].trim().to_string();
                current_file = Some(filename);
                current_content.clear();
            } else if current_file.is_some() {
                current_content.push(line);
            }
        }

        // Save last file
        if let Some(filename) = &current_file {
            let content = current_content.join("\n");
            if !content.trim().is_empty() {
                files.push((filename.clone(), content));
            }
        }

        files
    }

    /// Create basic project structure when AI response doesn't provide files
    fn create_basic_structure(&self, path: &Path, config: &ProjectConfig) -> Result<(), String> {
        // Create basic app.py
        let app_content = format!(
            r#"from fastapi import FastAPI
import uvicorn

app = FastAPI(title="{}", description="{}")

@app.get("/")
def read_root():
    return {{"message": "Hello from {}!"}}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)"#,
            config.project_type, config.description, config.project_type
        );

        fs_utils::write_file(&path.join("app.py"), app_content)
            .map_err(|e| format!("Failed to write app.py: {}", e))?;

        // Create requirements.txt
        let requirements = if config.dependencies.is_empty() {
            "fastapi\nuvicorn\npydantic".to_string()
        } else {
            config.dependencies.join("\n")
        };

        fs_utils::write_file(&path.join("requirements.txt"), requirements)
            .map_err(|e| format!("Failed to write requirements.txt: {}", e))?;

        // Create README.md
        let readme_content = format!(
            r#"# {}

{}

## Setup

```bash
pip install -r requirements.txt
python app.py
```

## Usage

Visit http://localhost:8000 to see the API."#,
            config.project_type, config.description
        );

        fs_utils::write_file(&path.join("README.md"), readme_content)
            .map_err(|e| format!("Failed to write README.md: {}", e))?;

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct ProjectConfig {
    pub python_version: String,
    pub project_type: String,
    pub dependencies: Vec<String>,
    pub description: String,
    pub selected_model: String,
}

impl Default for ProjectConfig {
    fn default() -> Self {
        Self {
            python_version: "3.11".to_string(),
            project_type: "General".to_string(),
            dependencies: Vec::new(),
            description: String::new(),
            selected_model: String::new(),
        }
    }
}
