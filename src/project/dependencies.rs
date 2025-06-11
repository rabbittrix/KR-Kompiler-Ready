use std::process::Command;

pub fn install_linting_tools(path: &std::path::Path, _py_version: &str) {
    let venv_path = path.join(".venv");
    let pip_binary = if cfg!(target_os = "windows") {
        venv_path.join("Scripts").join("pip")
    } else {
        venv_path.join("bin").join("pip")
    };

    let status = Command::new(&pip_binary)
        .arg("install")
        .arg("black")
        .arg("ruff")
        .arg("mypy")
        .status()
        .expect("Failed to execute pip command");

    if status.success() {
        println!("✅ Linting tools (Black, Ruff, MyPy) installed.");
    } else {
        eprintln!("❌ Failed to install linting tools.");
    }
}

pub fn install_testing_tools(path: &std::path::Path, _py_version: &str) {
    let venv_path = path.join(".venv");
    let pip_binary = if cfg!(target_os = "windows") {
        venv_path.join("Scripts").join("pip")
    } else {
        venv_path.join("bin").join("pip")
    };

    let status = Command::new(&pip_binary)
        .arg("install")
        .arg("pytest")
        .status()
        .expect("Failed to execute pip command");

    if status.success() {
        println!("✅ Testing framework (pytest) installed.");
    } else {
        eprintln!("❌ Failed to install pytest.");
    }
}