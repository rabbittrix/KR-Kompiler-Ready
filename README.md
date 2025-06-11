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

    🔌 Optional database installation (SQLite, Prisma)
    🧹 Linting tools (Black, Ruff, MyPy)
    🧪 Testing framework support (pytest)
    📁 Custom project structure generation
    🌐 GitHub template integration

🔨 Step 1: Project Setup in Rust

        Add dependencies in Cargo.toml:

            dialoguer for interactive prompts
            dirs for finding system paths
            which to check installed binaries
            clap for command-line argument parsing

🚀 Step 2: Commands to Test and Use the System

    Here are all the steps and terminal commands you need to test and run your new kr CLI tool.
    📦 1. Build the Project
            cargo build --release
            cargo build

    💻 2. Run the Tool
            cargo run -- new

    You’ll be prompted through:

        Project name
        Python version
        Project type

        Optional features:
            SQLite
            Prisma ORM
            Linting tools (Black, Ruff, MyPy)
            Testing tools (pytest)
            Git initialization

    🧪 3. Try Sample Inputs

        $ cargo run -- new
        🎨 Welcome to KR - Python Project Generator!
        What is the name of your project? my_project
        Choose Python version:
        > 3.11
        Choose project type:
        > API
        Install SQLite? [y/N] n
        Install Prisma ORM? [y/N] n
        Add linting tools (Black, Ruff, MyPy)? [Y/n] y
        Add testing framework (pytest)? [Y/n] y
        Initialize Git repo and push to GitHub? [y/N] n

    This will generate:

        my_project/
        ├── .venv/              # Virtual environment
        ├── app.py              # Flask API template
        ├── README.md
        ├── requirements.txt
        