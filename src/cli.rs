use clap::{Command, Arg};
use crate::project::scaffold::create_new_project;

pub fn run() {
    let matches = Command::new("kr")
        .version("0.1.0")
        .author("Your Name <you@example.com>")
        .about("KR: Python Project Manager")
        .subcommand(
            Command::new("new")
                .about("Create a new Python project")
                .arg(Arg::new("name")
                    .short('n')
                    .long("name")
                    .value_name("NAME")
                    .help("Sets the project name")
                    .required(false))
        )
        .get_matches();

    if let Some(_matches) = matches.subcommand_matches("new") {
        create_new_project();
    } else {
        println!("Use 'kr new' to create a new project.");
    }
}