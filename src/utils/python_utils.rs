use std::path::Path;
use std::process::Command;

pub fn python_exists(version: &str) -> bool {
    let output = Command::new(format!("python{}", version))
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();

    match output {
        Ok(mut child) => {
            let status = child.wait().unwrap();
            status.success()
        }
        Err(_) => false,
    }
}

pub fn install_python(version: &str) {
    #[cfg(target_os = "windows")]
    println!(
        "👉 Run: winget install -e --id Python.Python{} --exact",
        version.replace('.', "")
    );

    #[cfg(target_os = "macos")]
    println!("👉 Run: brew install python@{}", version);

    #[cfg(target_os = "linux")]
    println!("👉 Run: sudo apt install python{}-venv", version);
}

pub fn create_venv(path: &Path, py_version: &str) {
    let venv_path = path.join(".venv");
    println!(
        "⏳ Creating virtual environment with Python {}...",
        py_version
    );

    let mut cmd = Command::new(format!("python{}", py_version));
    let success = cmd
        .arg("-m")
        .arg("venv")
        .arg(&venv_path)
        .spawn()
        .is_ok_and(|mut child| child.wait().map_or(false, |status| status.success()));

    if !success {
        println!("⚠️ Versioned Python not found. Falling back to generic `python`...");

        let fallback_success = Command::new("python")
            .arg("-m")
            .arg("venv")
            .arg(&venv_path)
            .spawn()
            .is_ok_and(|mut child| child.wait().map_or(false, |status| status.success()));

        if !fallback_success {
            panic!("❌ Failed to create virtual environment. Is Python installed and in PATH?");
        }
    }

    // Verify the virtual environment was created successfully
    if verify_venv_created(path) {
        println!("✅ Virtual environment created at {:?}", venv_path);
    } else {
        panic!(
            "❌ Virtual environment was not created properly. Please check your Python installation."
        );
    }
}

pub fn upgrade_pip(path: &Path, _py_version: &str) {
    let venv_path = path.join(".venv");
    let pip_binary = if cfg!(target_os = "windows") {
        venv_path.join("Scripts").join("pip.exe")
    } else {
        venv_path.join("bin").join("pip")
    };

    // Verify pip is available
    if !verify_pip_available(path) {
        eprintln!("❌ pip is not available in the virtual environment. Skipping pip upgrade.");
        return;
    }

    println!("🔧 Upgrading pip...");

    // First try using pip binary directly
    let status = Command::new(&pip_binary)
        .arg("install")
        .arg("--upgrade")
        .arg("pip")
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("✅ pip upgraded successfully.");
            return;
        }
        _ => {
            println!("⚠️ Direct pip upgrade failed, trying alternative method...");
        }
    }

    // Fallback: use python -m pip
    let python_binary = if cfg!(target_os = "windows") {
        venv_path.join("Scripts").join("python.exe")
    } else {
        venv_path.join("bin").join("python")
    };

    let fallback_status = Command::new(&python_binary)
        .arg("-m")
        .arg("pip")
        .arg("install")
        .arg("--upgrade")
        .arg("pip")
        .status();

    match fallback_status {
        Ok(s) if s.success() => println!("✅ pip upgraded successfully (fallback method)."),
        _ => {
            eprintln!("❌ Failed to upgrade pip. Continuing with installation...");
            // Don't panic, just continue
        }
    }
}

/// Install dependencies from requirements.txt
pub fn install_requirements(path: &Path, _py_version: &str) {
    let venv_path = path.join(".venv");
    let pip_binary = if cfg!(target_os = "windows") {
        venv_path.join("Scripts").join("pip.exe")
    } else {
        venv_path.join("bin").join("pip")
    };

    let req_file = path.join("requirements.txt");

    if !req_file.exists() {
        println!("⚠️ requirements.txt not found. Skipping dependency installation.");
        return;
    }

    // Check if requirements.txt is empty
    if let Ok(content) = std::fs::read_to_string(&req_file) {
        if content.trim().is_empty() {
            println!("ℹ️ requirements.txt is empty. No dependencies to install.");
            return;
        }
    }

    // Verify pip is available
    if !verify_pip_available(path) {
        eprintln!(
            "❌ pip is not available in the virtual environment. Please ensure the virtual environment was created correctly."
        );
        return;
    }

    println!("🔧 Installing dependencies from requirements.txt...");

    // First attempt: direct pip install
    let status = Command::new(&pip_binary)
        .arg("install")
        .arg("-r")
        .arg("requirements.txt")
        .current_dir(path)
        .status();

    match status {
        Ok(s) if s.success() => {
            println!("✅ Dependencies installed successfully.");
            return;
        }
        _ => {
            println!("⚠️ Direct pip install failed, trying alternative method...");
        }
    }

    // Fallback: use python -m pip
    let python_binary = if cfg!(target_os = "windows") {
        venv_path.join("Scripts").join("python.exe")
    } else {
        venv_path.join("bin").join("python")
    };

    let fallback_status = Command::new(&python_binary)
        .arg("-m")
        .arg("pip")
        .arg("install")
        .arg("-r")
        .arg("requirements.txt")
        .current_dir(path)
        .status();

    match fallback_status {
        Ok(s) if s.success() => {
            println!("✅ Dependencies installed successfully (fallback method).")
        }
        _ => {
            eprintln!("❌ Failed to install dependencies. You may need to install them manually:");
            eprintln!("   cd {:?}", path);
            eprintln!("   .venv\\Scripts\\pip install -r requirements.txt  # Windows");
            eprintln!("   .venv/bin/pip install -r requirements.txt       # Unix");
        }
    }
}

pub fn install_sqlite(path: &Path, _py_version: &str) {
    let venv_path = path.join(".venv");
    let pip_binary = if cfg!(target_os = "windows") {
        venv_path.join("Scripts").join("pip.exe")
    } else {
        venv_path.join("bin").join("pip")
    };

    println!("🔧 Installing SQLite support...");

    // On Windows, pysqlite3-binary is often used instead of sqlite3
    let package = if cfg!(target_os = "windows") {
        "pysqlite3-binary"
    } else {
        "sqlite3"
    };

    let status = Command::new(&pip_binary)
        .arg("install")
        .arg(package)
        .status()
        .expect("Failed to execute pip command");

    if status.success() {
        println!("✅ SQLite support installed.");
    } else {
        eprintln!("❌ Failed to install SQLite.");
    }
}

pub fn install_prisma(path: &Path, _py_version: &str) {
    let venv_path = path.join(".venv");
    let pip_binary = if cfg!(target_os = "windows") {
        venv_path.join("Scripts").join("pip.exe")
    } else {
        venv_path.join("bin").join("pip")
    };

    println!("🔧 Installing Prisma ORM...");

    let status = Command::new(&pip_binary)
        .arg("install")
        .arg("prisma")
        .status()
        .expect("Failed to execute pip command");

    if status.success() {
        println!("✅ Prisma ORM installed.");
    } else {
        eprintln!("❌ Failed to install Prisma.");
    }
}

/// Verify that the virtual environment was created successfully
fn verify_venv_created(path: &Path) -> bool {
    let venv_path = path.join(".venv");

    if !venv_path.exists() {
        return false;
    }

    // Check for Python binary
    let python_binary = if cfg!(target_os = "windows") {
        venv_path.join("Scripts").join("python.exe")
    } else {
        venv_path.join("bin").join("python")
    };

    if !python_binary.exists() {
        return false;
    }

    // Test if Python works
    let status = Command::new(&python_binary)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();

    status.map_or(false, |s| s.success())
}

/// Verify that pip is available in the virtual environment
fn verify_pip_available(path: &Path) -> bool {
    let venv_path = path.join(".venv");
    let pip_binary = if cfg!(target_os = "windows") {
        venv_path.join("Scripts").join("pip.exe")
    } else {
        venv_path.join("bin").join("pip")
    };

    if !pip_binary.exists() {
        return false;
    }

    // Test if pip works
    let status = Command::new(&pip_binary)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();

    status.map_or(false, |s| s.success())
}
