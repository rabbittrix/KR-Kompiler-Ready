use std::fs;
use std::io;
use std::path::Path;

/// Creates a directory and all its parents if they don't exist.
pub fn create_dir<P: AsRef<Path>>(path: P) -> io::Result<()> {
    fs::create_dir_all(path.as_ref())
}

/// Writes content to a file.
pub fn write_file<P: AsRef<Path>, C: AsRef<[u8]>>(path: P, contents: C) -> io::Result<()> {
    fs::write(path.as_ref(), contents)
}