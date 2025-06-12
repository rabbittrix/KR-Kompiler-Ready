// scaffold.rs

use crate::project::dependencies;
use crate::project::templates;
use crate::utils::ai_utils::{OllamaAI, ProjectConfig};
use crate::utils::fs_utils;
use crate::utils::python_utils;
use dialoguer::{Confirm, Input, Select};
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

pub fn create_new_project() {
    println!("🎨 Welcome to KR - Python Project Generator!");

    // Ask for project name
    let name: String = Input::new()
        .with_prompt("What is the name of your project?")
        .interact_text()
        .unwrap();

    // Determine default path
    let default_path = std::env::current_dir().unwrap().join(&name);

    // Ask for custom path
    let input_path: String = Input::new()
        .with_prompt("Where should the project be created?")
        .default(default_path.to_str().unwrap().to_string())
        .interact_text()
        .unwrap();

    // Clean and validate the path
    let cleaned_path = input_path.trim().replace('"', "").replace("'", "");
    let mut path = PathBuf::from(&cleaned_path);

    // Convert relative path to absolute if needed
    if path.is_relative() {
        path = std::env::current_dir().unwrap().join(&path);
    }

    // Validate path for Windows compatibility
    if cfg!(target_os = "windows") {
        if let Some(path_str) = path.to_str() {
            // Check for invalid Windows characters (excluding drive letter colons)
            let invalid_chars = ['<', '>', '"', '|', '?', '*'];

            // Simple check: if path contains any invalid chars (excluding drive letter)
            let has_invalid_chars = invalid_chars.iter().any(|&c| path_str.contains(c));

            // Check for colons that are not in drive letter position
            let has_invalid_colon = {
                let parts: Vec<&str> = path_str.split('\\').collect();
                parts
                    .iter()
                    .any(|part| part.contains(':') && !part.ends_with(':'))
            };

            if has_invalid_chars || has_invalid_colon {
                println!(
                    "❌ Invalid characters in path. Windows paths cannot contain: < > : \" | ? *"
                );
                println!("💡 Please use a valid path without these characters.");
                println!("💡 Note: Drive letters like 'D:' are allowed.");
                return;
            }
        }
    }

    // Check if path already exists
    if path.exists() {
        if !Confirm::new()
            .with_prompt("Directory already exists. Continue?")
            .default(false)
            .interact()
            .unwrap()
        {
            println!("❌ Project creation canceled.");
            return;
        }
    }

    // Choose Python version (always prompt the user)
    let versions = vec!["3.9", "3.10", "3.11", "3.12", "3.13"];
    let selection = Select::new()
        .with_prompt("Choose Python version:")
        .default(3)
        .items(&versions)
        .interact()
        .unwrap();
    let py_version = versions[selection].to_string();

    // Check if Python exists
    if !python_utils::python_exists(&py_version) {
        println!("⏳ Installing Python {}...", py_version);
        python_utils::install_python(&py_version);
    }

    // Ask if user wants to use AI
    let use_ai = Confirm::new()
        .with_prompt("Do you have a localhost AI (like Ollama) and would you like to use it to help create the project?")
        .default(false)
        .interact()
        .unwrap();

    let (proj_type, ai_generated_config) = if use_ai {
        let (_, proj_type, config) = handle_ai_project_generation(&name, &py_version);
        (proj_type, config)
    } else {
        let (_, proj_type, config) = handle_manual_project_generation(&py_version);
        (proj_type, config)
    };

    // Create folder structure at the user-defined path
    match fs_utils::create_dir(&path) {
        Ok(_) => println!("📁 Project directory created successfully"),
        Err(e) => {
            println!("❌ Failed to create project directory: {}", e);
            println!("💡 Please check the path and try again.");
            return;
        }
    }

    // Show final selected path
    println!("📁 Creating project at: {:?}", path);

    // Handle different AI modes
    if let Some(config) = Some(&ai_generated_config) {
        match config.project_type.as_str() {
            "Interactive" => {
                // Handle interactive AI session
                let ai = OllamaAI::new();
                let selected_model = &config.selected_model;

                match ai.interactive_project_creation(selected_model, &name, &path) {
                    Ok(final_config) => {
                        println!("🎉 Interactive AI project creation completed!");
                        println!("🤖 Final configuration:");
                        println!("  Python Version: {}", final_config.python_version);
                        println!("  Project Type: {}", final_config.project_type);
                        println!("  Dependencies: {}", final_config.dependencies.join(", "));

                        // Create virtual environment and install dependencies
                        python_utils::create_venv(&path, &final_config.python_version);
                        python_utils::install_requirements(&path, &final_config.python_version);
                        python_utils::upgrade_pip(&path, &final_config.python_version);

                        // If AI generated config, create custom requirements.txt
                        create_ai_generated_requirements(&path, config);

                        println!(
                            "🎉 Project '{}' created successfully with AI assistance!",
                            name
                        );
                        ensure_requirements_from_imports(&path);
                        ensure_requirements_from_pip_freeze(&path, &py_version);
                        return;
                    }
                    Err(e) => {
                        println!("❌ Interactive AI session failed: {}", e);
                        println!("💡 Falling back to template-based creation...");

                        // Actually create the template files here
                        // Create virtual environment
                        python_utils::create_venv(&path, &py_version);

                        // Write README.md
                        fs_utils::write_file(
                            &path.join("README.md"),
                            format!("# {}\n\nGenerated using KR - Python Project Manager (AI fallback).", name),
                        )
                        .expect("Failed to write README.md");

                        // Handle optional features
                        handle_optional_features(&path, &py_version);

                        // Create full template structure based on project type
                        match proj_type.as_str() {
                            "Hello World" => {
                                // These use the simple app.py structure
                                let content_vec = templates::get_app_content();
                                let content = content_vec
                                    .into_iter()
                                    .map(|(_, content)| content)
                                    .collect::<Vec<&str>>()
                                    .join("\n");
                                fs_utils::write_file(&path.join("app.py"), content)
                                    .expect("Failed to write app.py");
                            }
                            "FastAPI Simple" => {
                                // Use the simple FastAPI template with app.py
                                templates::create_template_structure(&path, &proj_type);
                            }
                            _ => {
                                // All other types use structured template
                                templates::create_template_structure(&path, &proj_type);
                            }
                        }

                        // If AI generated config, create custom requirements.txt
                        create_ai_generated_requirements(&path, config);

                        // Install requirements first (before upgrading pip)
                        python_utils::install_requirements(&path, &py_version);

                        // Then upgrade pip (after requirements are installed)
                        python_utils::upgrade_pip(&path, &py_version);

                        println!(
                            "🎉 Project '{}' created successfully with template fallback!",
                            name
                        );
                        ensure_requirements_from_imports(&path);
                        ensure_requirements_from_pip_freeze(&path, &py_version);
                        return;
                    }
                }
            }
            "AI Generated" => {
                // Handle quick AI generation
                let ai = OllamaAI::new();
                let selected_model = &config.selected_model;

                match ai.generate_project_files(selected_model, &name, &config.description, &path) {
                    Ok(final_config) => {
                        println!("🎉 AI-generated project created successfully!");

                        // Create virtual environment and install dependencies
                        python_utils::create_venv(&path, &final_config.python_version);
                        python_utils::install_requirements(&path, &final_config.python_version);
                        python_utils::upgrade_pip(&path, &final_config.python_version);

                        // If AI generated config, create custom requirements.txt
                        create_ai_generated_requirements(&path, config);

                        println!(
                            "🎉 Project '{}' created successfully with AI assistance!",
                            name
                        );
                        ensure_requirements_from_imports(&path);
                        ensure_requirements_from_pip_freeze(&path, &py_version);
                        return;
                    }
                    Err(e) => {
                        println!("❌ AI file generation failed: {}", e);
                        println!("💡 Falling back to manual (template-based) creation...");

                        // Switch to manual mode and get manual config
                        let (_, manual_proj_type, manual_config) =
                            handle_manual_project_generation(&py_version);

                        // Create virtual environment
                        python_utils::create_venv(&path, &py_version);

                        // Write README.md
                        fs_utils::write_file(
                            &path.join("README.md"),
                            format!("# {}\n\n{}\n\nGenerated using KR - Python Project Manager (AI fallback).", name, manual_config.description),
                        )
                        .expect("Failed to write README.md");

                        // Handle optional features
                        handle_optional_features(&path, &py_version);

                        // Create full template structure based on manual project type
                        match manual_proj_type.as_str() {
                            "Hello World" => {
                                let content_vec = templates::get_app_content();
                                let content = content_vec
                                    .into_iter()
                                    .map(|(_, content)| content)
                                    .collect::<Vec<&str>>()
                                    .join("\n");
                                fs_utils::write_file(&path.join("app.py"), content)
                                    .expect("Failed to write app.py");
                            }
                            "FastAPI Simple" => {
                                templates::create_template_structure(&path, &manual_proj_type);
                            }
                            _ => {
                                templates::create_template_structure(&path, &manual_proj_type);
                            }
                        }

                        // Create requirements.txt
                        create_ai_generated_requirements(&path, &manual_config);

                        // Install requirements first (before upgrading pip)
                        python_utils::install_requirements(&path, &py_version);
                        python_utils::upgrade_pip(&path, &py_version);

                        println!(
                            "🎉 Project '{}' created successfully with manual fallback!",
                            name
                        );
                        ensure_requirements_from_imports(&path);
                        ensure_requirements_from_pip_freeze(&path, &py_version);
                        return;
                    }
                }
            }
            _ => {
                // Handle AI-assisted configuration (original behavior)
                // Create virtual environment
                python_utils::create_venv(&path, &py_version);

                // Write README.md
                let readme_content = format!(
                    "# {}\n\n{}\n\nGenerated using KR - Python Project Manager with AI assistance.",
                    name, config.description
                );

                fs_utils::write_file(&path.join("README.md"), readme_content)
                    .expect("Failed to write README.md");

                // Handle optional features
                handle_optional_features(&path, &py_version);

                // Create full template structure based on project type
                match proj_type.as_str() {
                    "Hello World" => {
                        // These use the simple app.py structure
                        let content_vec = templates::get_app_content();
                        let content = content_vec
                            .into_iter()
                            .map(|(_, content)| content)
                            .collect::<Vec<&str>>()
                            .join("\n");
                        fs_utils::write_file(&path.join("app.py"), content)
                            .expect("Failed to write app.py");
                    }
                    "FastAPI Simple" => {
                        // Use the simple FastAPI template with app.py
                        templates::create_template_structure(&path, &proj_type);
                    }
                    _ => {
                        // All other types use structured template
                        templates::create_template_structure(&path, &proj_type);
                    }
                }

                // If AI generated config, create custom requirements.txt
                create_ai_generated_requirements(&path, config);

                // Install requirements first (before upgrading pip)
                python_utils::install_requirements(&path, &py_version);

                // Then upgrade pip (after requirements are installed)
                python_utils::upgrade_pip(&path, &py_version);

                println!("🎉 Project '{}' created successfully!", name);
                ensure_requirements_from_imports(&path);
                ensure_requirements_from_pip_freeze(&path, &py_version);
                return;
            }
        }
    }

    // Fallback to manual creation if AI modes failed or weren't used
    // Create virtual environment
    python_utils::create_venv(&path, &py_version);

    // Write README.md
    fs_utils::write_file(
        &path.join("README.md"),
        format!("# {}\n\nGenerated using KR - Python Project Manager.", name),
    )
    .expect("Failed to write README.md");

    // Handle optional features
    handle_optional_features(&path, &py_version);

    // Create full template structure based on project type
    match proj_type.as_str() {
        "Hello World" => {
            // These use the simple app.py structure
            let content_vec = templates::get_app_content();
            let content = content_vec
                .into_iter()
                .map(|(_, content)| content)
                .collect::<Vec<&str>>()
                .join("\n");
            fs_utils::write_file(&path.join("app.py"), content).expect("Failed to write app.py");
        }
        "FastAPI Simple" => {
            // Use the simple FastAPI template with app.py
            templates::create_template_structure(&path, &proj_type);
        }
        _ => {
            // All other types use structured template
            templates::create_template_structure(&path, &proj_type);
        }
    }

    // Install requirements first (before upgrading pip)
    python_utils::install_requirements(&path, &py_version);

    // Then upgrade pip (after requirements are installed)
    python_utils::upgrade_pip(&path, &py_version);

    println!("🎉 Project '{}' created successfully!", name);
    ensure_requirements_from_imports(&path);
    ensure_requirements_from_pip_freeze(&path, &py_version);
}

fn handle_ai_project_generation(
    _project_name: &str,
    py_version: &str,
) -> (String, String, ProjectConfig) {
    let ai = OllamaAI::new();

    // Check if Ollama is available
    if !ai.is_available() {
        println!("❌ Ollama is not running or not accessible at http://localhost:11434");
        println!("💡 Please start Ollama and try again, or continue without AI.");

        if Confirm::new()
            .with_prompt("Continue without AI?")
            .default(true)
            .interact()
            .unwrap()
        {
            return handle_manual_project_generation(py_version);
        } else {
            println!("❌ Project creation canceled.");
            std::process::exit(1);
        }
    }

    // Get available models
    let models = match ai.list_models() {
        Ok(models) => models,
        Err(e) => {
            println!("❌ Failed to get AI models: {}", e);
            println!("💡 Continuing without AI...");
            return handle_manual_project_generation(py_version);
        }
    };

    if models.is_empty() {
        println!(
            "❌ No AI models found. Please install some models with 'ollama pull <model_name>'"
        );
        println!("💡 Continuing without AI...");
        return handle_manual_project_generation(py_version);
    }

    // Display available models
    println!("\n🤖 Available AI Models:");
    for (i, model) in models.iter().enumerate() {
        println!("  {}. {}", i + 1, model);
    }

    // Let user select model
    let model_selection = Select::new()
        .with_prompt("Which AI model would you like to use?")
        .items(&models)
        .interact()
        .unwrap();

    let selected_model = &models[model_selection];

    // Ask user what type of AI assistance they want
    let ai_options = vec![
        "Interactive AI session (ask questions, get guidance)",
        "Quick AI generation (describe project, get complete files)",
        "AI-assisted configuration only (suggestions only)",
    ];

    let ai_option = Select::new()
        .with_prompt("What type of AI assistance would you like?")
        .items(&ai_options)
        .interact()
        .unwrap();

    match ai_option {
        0 => {
            // Interactive AI session
            println!("🤖 Starting interactive AI session...");
            println!("💬 The AI will help you create your project step by step.");
            println!("💬 You can ask questions, request changes, or ask for clarification.");

            // We'll handle the interactive session in the main function
            // For now, return a placeholder that indicates interactive mode
            let mut config = ProjectConfig::default();
            config.python_version = py_version.to_string();
            config.project_type = "Interactive".to_string();
            config.description = "Interactive AI-assisted project".to_string();
            config.selected_model = selected_model.to_string();

            (
                config.python_version.clone(),
                config.project_type.clone(),
                config,
            )
        }
        1 => {
            // Quick AI generation
            let project_description: String = Input::new()
                .with_prompt(
                    "Describe your project (what should it do? what features do you need?)",
                )
                .interact_text()
                .unwrap();

            println!("🤖 Generating complete project with {}...", selected_model);

            // We'll handle the file generation in the main function
            let mut config = ProjectConfig::default();
            config.python_version = py_version.to_string();
            config.project_type = "AI Generated".to_string();
            config.description = project_description.clone();
            config.selected_model = selected_model.to_string();

            (
                config.python_version.clone(),
                config.project_type.clone(),
                config,
            )
        }
        2 => {
            // AI-assisted configuration only (original behavior)
            let project_description: String = Input::new()
                .with_prompt(
                    "Describe your project (what should it do? what features do you need?)",
                )
                .interact_text()
                .unwrap();

            println!("🤖 Generating project structure with {}...", selected_model);

            // Generate project with AI
            match ai.generate_project(selected_model, &project_description) {
                Ok(ai_response) => {
                    println!("✅ AI generated project structure successfully!");

                    // Parse AI response with enhanced parsing
                    let config = ai.parse_ai_response(&ai_response);

                    // Validate the configuration
                    if let Err(validation_error) = ai.validate_config(&config) {
                        println!(
                            "⚠️ AI configuration validation failed: {}",
                            validation_error
                        );
                        println!("💡 Using fallback configuration...");
                        return handle_manual_project_generation(py_version);
                    }

                    // Show AI suggestions with better formatting
                    println!("\n🤖 AI Suggestions:");
                    println!("  📦 Python Version: {}", config.python_version);
                    println!("  🏗️  Project Type: {}", config.project_type);
                    println!("  📚 Dependencies: {}", config.dependencies.join(", "));
                    if !config.description.is_empty() {
                        println!("  📝 Description: {}", config.description);
                    }

                    // Ask user to confirm or modify
                    if Confirm::new()
                        .with_prompt("Use AI-generated configuration?")
                        .default(true)
                        .interact()
                        .unwrap()
                    {
                        (
                            config.python_version.clone(),
                            config.project_type.clone(),
                            config,
                        )
                    } else {
                        println!("💡 Using manual configuration instead...");
                        handle_manual_project_generation(py_version)
                    }
                }
                Err(e) => {
                    let error_message = ai.handle_ai_error(&e);
                    println!("{}", error_message);
                    println!("💡 Continuing with manual configuration...");
                    handle_manual_project_generation(py_version)
                }
            }
        }
        _ => handle_manual_project_generation(py_version),
    }
}

fn handle_manual_project_generation(py_version: &str) -> (String, String, ProjectConfig) {
    // Choose project type
    let types = vec![
        "Hello World",
        "API",
        "FastAPI",
        "FastAPI Simple",
        "Modular",
        "Microservices",
        "AI",
        "ML",
        "DL",
        "Django",
        "Streamlit",
        "Streamlit + Docling + LangChain",
    ];
    let proj_type_idx = Select::new()
        .with_prompt("Choose project type:")
        .items(&types)
        .interact()
        .unwrap();
    let proj_type = types[proj_type_idx].to_string();

    (py_version.to_string(), proj_type, ProjectConfig::default())
}

fn create_ai_generated_requirements(path: &Path, config: &ProjectConfig) {
    let requirements_content = config.dependencies.join("\n");
    fs_utils::write_file(&path.join("requirements.txt"), requirements_content)
        .expect("Failed to write AI-generated requirements.txt");

    println!("📝 Created requirements.txt with AI-suggested dependencies");
}

fn handle_optional_features(path: &std::path::Path, py_version: &str) {
    if Confirm::new()
        .with_prompt("Install SQLite?")
        .default(false)
        .interact()
        .unwrap()
    {
        python_utils::install_sqlite(path, py_version);
    }

    if Confirm::new()
        .with_prompt("Install Prisma ORM?")
        .default(false)
        .interact()
        .unwrap()
    {
        python_utils::install_prisma(path, py_version);
    }

    if Confirm::new()
        .with_prompt("Add linting tools (Black, Ruff, MyPy)?")
        .default(true)
        .interact()
        .unwrap()
    {
        dependencies::install_linting_tools(path, py_version);
    }

    if Confirm::new()
        .with_prompt("Add testing framework (pytest)?")
        .default(true)
        .interact()
        .unwrap()
    {
        dependencies::install_testing_tools(path, py_version);
    }

    if Confirm::new()
        .with_prompt("Initialize Git repo and push to GitHub?")
        .default(false)
        .interact()
        .unwrap()
    {
        crate::utils::git_utils::init_git_repo(&path);
    }
}

#[allow(dead_code)]
fn install_flask(path: &Path, _py_version: &str) {
    let venv_path = path.join(".venv");
    let pip_binary = if cfg!(target_os = "windows") {
        venv_path.join("Scripts").join("pip.exe")
    } else {
        venv_path.join("bin").join("pip")
    };

    println!("🔧 Installing Flask...");

    let status = Command::new(pip_binary)
        .arg("install")
        .arg("flask")
        .status()
        .expect("Failed to execute pip command");

    if status.success() {
        println!("✅ Flask installed.");
    } else {
        eprintln!("❌ Failed to install Flask.");
    }
}

/// Ensures requirements.txt is generated from imports if missing or empty
fn ensure_requirements_from_imports(project_path: &Path) {
    let mut packages = HashSet::new();
    if let Ok(entries) = fs::read_dir(project_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "py").unwrap_or(false) {
                if let Ok(content) = fs::read_to_string(&path) {
                    for line in content.lines() {
                        if line.trim_start().starts_with("import ")
                            || line.trim_start().starts_with("from ")
                        {
                            if let Some(pkg) = line.split_whitespace().nth(1) {
                                let base = pkg.split('.').next().unwrap_or(pkg);
                                packages.insert(base.to_string());
                            }
                        }
                    }
                }
            }
        }
    }
    // Map common packages to PyPI names
    let mut reqs = Vec::new();
    for pkg in packages {
        let pypi = match pkg.as_str() {
            "fastapi" => "fastapi",
            "uvicorn" => "uvicorn",
            "pydantic" => "pydantic",
            "flask" => "flask",
            "sqlalchemy" => "sqlalchemy",
            "pytest" => "pytest",
            "requests" => "requests",
            "numpy" => "numpy",
            "pandas" => "pandas",
            "scipy" => "scipy",
            "sklearn" => "scikit-learn",
            "matplotlib" => "matplotlib",
            "seaborn" => "seaborn",
            _ => pkg.as_str(),
        };
        reqs.push(pypi.to_string());
    }
    let reqs_path = project_path.join("requirements.txt");
    let needs_write = !reqs_path.exists()
        || fs::read_to_string(&reqs_path)
            .map(|c| c.trim().is_empty())
            .unwrap_or(true);
    if needs_write && !reqs.is_empty() {
        let content = reqs.join("\n");
        fs::write(&reqs_path, content).expect("Failed to write requirements.txt from imports");
        println!("📝 requirements.txt generated from imports.");
    }
}

/// Ensures requirements.txt is filled by running pip freeze if still empty
fn ensure_requirements_from_pip_freeze(project_path: &Path, _py_version: &str) {
    let reqs_path = project_path.join("requirements.txt");
    let is_empty = fs::read_to_string(&reqs_path)
        .map(|c| c.trim().is_empty())
        .unwrap_or(true);
    if is_empty {
        println!("⚠️ requirements.txt is still empty. Attempting to fill with 'pip freeze'...");
        let venv_path = project_path.join(".venv");
        let pip_bin = if cfg!(target_os = "windows") {
            venv_path.join("Scripts").join("pip.exe")
        } else {
            venv_path.join("bin").join("pip")
        };
        let output = Command::new(pip_bin).arg("freeze").output();
        if let Ok(output) = output {
            if output.status.success() {
                let freeze = String::from_utf8_lossy(&output.stdout);
                if !freeze.trim().is_empty() {
                    fs::write(&reqs_path, freeze.as_ref())
                        .expect("Failed to write requirements.txt from pip freeze");
                    println!("📝 requirements.txt filled using pip freeze.");
                } else {
                    println!("⚠️ pip freeze did not return any packages.");
                }
            } else {
                println!("❌ Failed to run pip freeze to fill requirements.txt");
            }
        } else {
            println!("❌ Could not execute pip freeze to fill requirements.txt");
        }
    }
}
