use std::process::Command;
use std::path::Path;

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
    println!("👉 Run: winget install -e --id Python.Python{} --exact", version.replace('.', ""));

    #[cfg(target_os = "macos")]
    println!("👉 Run: brew install python@{}", version);

    #[cfg(target_os = "linux")]
    println!("👉 Run: sudo apt install python{}-venv", version);
}

pub fn create_venv(path: &Path, py_version: &str) {
    let venv_path = path.join(".venv");
    println!("⏳ Creating virtual environment with Python {}...", py_version);

    let mut cmd = Command::new(format!("python{}", py_version));
    let success = cmd.arg("-m").arg("venv").arg(&venv_path).spawn().is_ok_and(|mut child| {
        child.wait().map_or(false, |status| status.success())
    });

    if !success {
        println!("⚠️ Versioned Python not found. Falling back to generic `python`...");

        let fallback_success = Command::new("python")
            .arg("-m")
            .arg("venv")
            .arg(&venv_path)
            .spawn()
            .is_ok_and(|mut child| {
                child.wait().map_or(false, |status| status.success())
            });

        if !fallback_success {
            panic!("❌ Failed to create virtual environment. Is Python installed and in PATH?");
        }
    }

    println!("✅ Virtual environment created at {:?}", venv_path);
}

pub fn install_sqlite(path: &Path, _py_version: &str) {
    let venv_path = path.join(".venv");
    let pip_binary = if cfg!(target_os = "windows") {
        venv_path.join("Scripts").join("pip")
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
        venv_path.join("Scripts").join("pip")
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

