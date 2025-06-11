pub fn get_app_content(proj_type: &str) -> &'static str {
    match proj_type {
        "Hello World" => "print('Hello from KR!')\n",
        "API" => "from flask import Flask\n\napp = Flask(__name__)\n\n@app.route('/')\ndef hello():\n    return 'Hello API'\n\nif __name__ == '__main__':\n    app.run()",
        "Django" => "# Run django-admin startproject after installing Django",
        _ => "",
    }
}

#[allow(dead_code)]
pub fn get_requirements(proj_type: &str) -> String {
    match proj_type {
        "Hello World" => "streamlit\n".to_string(),
        "API" => "flask\n".to_string(),
        "Django" => "django\n".to_string(),
        _ => "".to_string(),
    }
}