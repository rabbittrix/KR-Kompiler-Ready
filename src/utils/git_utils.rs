use std::process::Command;
use std::path::Path;

pub fn init_git_repo(path: &Path) {
    Command::new("git")
        .arg("init")
        .current_dir(path)
        .spawn()
        .expect("Git init failed")
        .wait()
        .expect("Git init failed");

    Command::new("git")
        .arg("add")
        .arg(".")
        .current_dir(path)
        .spawn()
        .expect("Git add failed")
        .wait()
        .expect("Git add failed");

    Command::new("git")
        .arg("commit")
        .arg("-m")
        .arg("Initial commit by KR")
        .current_dir(path)
        .spawn()
        .expect("Git commit failed")
        .wait()
        .expect("Git commit failed");

    println!("Git repo initialized and committed.");
}