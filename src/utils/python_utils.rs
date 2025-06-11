use std::process::Command;
use std::path::Path;

pub fn python_exists(version: &str) -> bool {
    match which::which(format!("python{}", version)) {
        Ok(_) => true,
        Err(_) => false,
    }
}

pub fn install_python(version: &str) {
    #[cfg(target_os = "windows")]
    println!("👉 Run: winget install -e --id Python.Python.{} --exact", version.replace('.', ""));

    #[cfg(target_os = "macos")]
    println!("👉 Run: brew install python@{}", version);

    #[cfg(target_os = "linux")]
    println!("👉 Run: sudo apt install python{}-venv", version);
}

pub fn create_venv(path: &Path, py_version: &str) {
    let venv_path = path.join(".venv");
    println!("⏳ Creating virtual environment with Python {}...", py_version);

    Command::new(format!("python{}", py_version))
        .arg("-m")
        .arg("venv")
        .arg(&venv_path)
        .spawn()
        .expect("Failed to create virtual environment")
        .wait()
        .expect("Virtual env creation failed");

    println!("✅ Virtual environment created at {:?}", venv_path);
}

pub fn install_sqlite(path: &Path, _py_version: &str) {
    let venv_path = path.join(".venv");
    let pip = venv_path.join("Scripts").join("pip");

    Command::new(pip)
        .arg("install")
        .arg("sqlite3")
        .spawn()
        .expect("Failed to install SQLite")
        .wait()
        .expect("SQLite install failed");
}

pub fn install_prisma(path: &Path, _py_version: &str) {
    let venv_path = path.join(".venv");
    let pip = venv_path.join("Scripts").join("pip");

    Command::new(pip)
        .arg("install")
        .arg("prisma")
        .spawn()
        .expect("Failed to install Prisma")
        .wait()
        .expect("Prisma install failed");
}

