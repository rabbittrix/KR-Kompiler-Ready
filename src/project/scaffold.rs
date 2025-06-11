use dialoguer::{Input, Select, Confirm};
use std::path::PathBuf;
use crate::project::dependencies;
use crate::project::templates;
use crate::utils::fs_utils;
use crate::utils::python_utils;

pub fn create_new_project() {
    println!("🎨 Welcome to KR - Python Project Generator!");

    // Ask for project name
    let name: String = Input::new()
        .with_prompt("What is the name of your project?")
        .interact_text()
        .unwrap();

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
        "Modular",
        "Microservices",
        "AI",
        "ML",
        "DL",
        "Django",
    ];
    let proj_type_idx = Select::new()
        .with_prompt("Choose project type:")
        .items(&types)
        .interact()
        .unwrap();
    let proj_type = types[proj_type_idx];

    // Create folder structure
    let path = PathBuf::from(&name);
    fs_utils::create_dir(&path).expect("Failed to create project directory");

    python_utils::create_venv(&path, py_version);
    fs_utils::write_file(&path.join("app.py"), templates::get_app_content(proj_type))
        .expect("Failed to write app.py file");
    fs_utils::write_file(&path.join("README.md"), format!("# {}\n\nGenerated using KR - Python Project Manager.", name))
        .expect("Failed to write README.md file");

    // Handle optional features
    handle_optional_features(&path, py_version);

    println!("✅ Project '{}' created successfully!", name);
}

fn handle_optional_features(path: &std::path::Path, py_version: &str) {
    // Optional DB install
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

    // Linting
    if Confirm::new()
        .with_prompt("Add linting tools (Black, Ruff, MyPy)?")
        .default(true)
        .interact()
        .unwrap()
    {
        dependencies::install_linting_tools(path, py_version);
    }

    // Testing
    if Confirm::new()
        .with_prompt("Add testing framework (pytest)?")
        .default(true)
        .interact()
        .unwrap()
    {
        dependencies::install_testing_tools(path, py_version);
    }

    // GitHub Integration
    if Confirm::new()
        .with_prompt("Initialize Git repo and push to GitHub?")
        .default(false)
        .interact()
        .unwrap()
    {
        crate::utils::git_utils::init_git_repo(path);
    }
}