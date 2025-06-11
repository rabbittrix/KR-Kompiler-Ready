# 🧱 Project Overview: KR (Kompiler Ready) - Python Project Manager

🎯 Goal:
    A CLI tool written in Rust that helps users create and manage Python projects with various templates and dependency management.---

🧪 Future Enhancements

    - GUI mode via Tauri
    - Integration with Poetry or Hatch
    - Multi-language support (Java, Node.js?)
    - Auto-generate CI/CD pipelines (.github/workflows/)
    - Git initialization and commit hooks
    
📁 Final Folder Structure of kr Rust CLI Tool

    kr/
    ├── Cargo.toml
    ├── src/
    │   ├── main.rs
    │   ├── cli.rs
    │   ├── project/
    │   │   ├── mod.rs
    │   │   ├── scaffold.rs
    │   │   ├── templates.rs
    │   │   └── dependencies.rs
    │   ├── utils/
    │   │   ├── fs_utils.rs
    │   │   ├── python_utils.rs
    │   │   └── git_utils.rs
    │   └── config/
    │       └── constants.rs
    ├── README.md
    └── .gitignore

🔨 Step 1: Project Setup in Rust

        Add dependencies in Cargo.toml:

            dialoguer for interactive prompts
            dirs for finding system paths
            which to check installed binaries
            clap for command-line argument parsing
            