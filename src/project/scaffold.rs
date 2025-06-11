// scaffold.rs

use crate::project::dependencies;
use crate::project::templates;
use crate::utils::fs_utils;
use crate::utils::python_utils;
use dialoguer::{Confirm, Input, Select};
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

    let path = PathBuf::from(input_path);

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

    // Choose Python version
    let versions = vec!["3.9", "3.10", "3.11", "3.12", "3.13"];
    let selection = Select::new()
        .with_prompt("Choose Python version:")
        .default(3)
        .items(&versions)
        .interact()
        .unwrap();
    let py_version = versions[selection];

    // Check if Python exists
    if !python_utils::python_exists(py_version) {
        println!("⏳ Installing Python {}...", py_version);
        python_utils::install_python(py_version);
    }

    // Choose project type
    let types = vec![
        "Hello World",
        "API",
        "FastAPI",
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
    let proj_type = types[proj_type_idx];

    // Create folder structure at the user-defined path
    fs_utils::create_dir(&path).expect("Failed to create project directory");

    // Show final selected path
    println!("📁 Creating project at: {:?}", path);

    // Create virtual environment
    python_utils::create_venv(&path, py_version);

    // Write README.md
    fs_utils::write_file(
        &path.join("README.md"),
        format!("# {}\n\nGenerated using KR - Python Project Manager.", name),
    )
    .expect("Failed to write README.md");

    // Handle optional features
    handle_optional_features(&path, py_version);

    // Create full template structure based on project type
    match proj_type {
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
        _ => {
            // All other types use structured template
            templates::create_template_structure(&path, proj_type);
        }
    }

    python_utils::upgrade_pip(&path, py_version);

    python_utils::install_requirements(&path, py_version); // Critical!

    python_utils::upgrade_pip(&path, py_version); // Optional but recommended

    println!("🎉 Project '{}' created successfully!", name);
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
        venv_path.join("Scripts").join("pip")
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
