#[macro_use] extern crate rocket;

use rocket::http::{Status, ContentType};
use rocket::serde::json::Json;
use serde::{Deserialize, Serialize};
use reqwest::Client;
use dotenv::dotenv;
use once_cell::sync::Lazy;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use std::io::{self, BufReader, BufWriter};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::net::IpAddr;
use std::env;
use std::fs::File;

#[derive(Deserialize, Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    conversation_id: String,
}

impl ChatRequest {
    fn validate(&self) -> Result<(), String> {
        if self.model.trim().is_empty() {
            return Err("Model cannot be empty.".to_string());
        }

        if self.messages.is_empty() {
            return Err("Message list cannot be empty.".to_string());
        }

        for message in &self.messages {
            if message.role.trim().is_empty() {
                return Err("Message role cannot be empty.".to_string());
            }

            if message.content.trim().is_empty() {
                return Err("Message content cannot be empty.".to_string());
            }
        }

        if self.conversation_id.trim().is_empty() {
            return Err("Conversation ID cannot be empty.".to_string());
        }

        Ok(())
    }
}

#[derive(Deserialize, Serialize, Default)]
struct TokenData {
    total_input_tokens: usize,
    total_output_tokens: usize,
    ip_token_counts: HashMap<String, (usize, usize)>,
}

#[derive(Deserialize, Serialize)]
pub struct Model {
    pub id: String,
    pub name: String,
    pub description: String,
}

#[derive(Deserialize)]
struct ModelsFile {
    models: Vec<Model>,
}

const RATE_LIMIT: u32 = 100;
const RATE_LIMIT_TIME_WINDOW: Duration = Duration::from_secs(60);

static REQUEST_COUNTS: Lazy<Mutex<HashMap<IpAddr, (u32, Instant)>>> = Lazy::new(|| Mutex::new(HashMap::new()));

#[get("/")]
fn index() -> (Status, (ContentType, String))  {
    let response = serde_json::json!({
            "message": "Hello from luuma!"
    });
    (Status::Ok, (ContentType::JSON, response.to_string()))
}

#[catch(404)]
fn not_found() -> (Status, (ContentType, String))  {
    let response = serde_json::json!({
            "message": "The requested resource was not found."
        });
    (Status::NotFound, (ContentType::JSON, response.to_string()))
}

#[get("/models")]
fn models() -> (Status, (ContentType, String)) {
    let models = load_models();
    let response = serde_json::json!(models);
    (Status::Ok, (ContentType::JSON, response.to_string()))
}

fn load_models() -> Vec<Model> {
    let file = File::open("models.json").expect("Cannot open models.json file");
    let reader = BufReader::new(file);
    let models_file: ModelsFile = serde_json::from_reader(reader).expect("Error reading JSON data");
    models_file.models
}

#[post("/conversations/messages", data = "<chat_request>")]
async fn message(chat_request: Json<ChatRequest>, client_ip: Option<IpAddr>) -> (Status, (ContentType, String)) {
    if let Some(ip) = client_ip {
        if !check_rate_limit(ip) {
            return (Status::TooManyRequests, (ContentType::JSON, "{\"error\": \"Too Many Requests\"}".to_string()));
        }
    } else {
        return (Status::BadRequest, (ContentType::JSON, "{\"error\": \"Unable to determine client IP\"}".to_string()));
    }

    if let Err(validation_error) = chat_request.validate() {
        return (Status::BadRequest, (ContentType::JSON, serde_json::json!({"error": validation_error}).to_string()));
    }
    
    let input_tokens = count_tokens(&chat_request.messages);
    
    let request_body = serde_json::json!({
        "messages": chat_request.messages,
        "model": chat_request.model,
        "stream": false
    });

    let client = Client::new();
    let response = client.post("https://api.groq.com/openai/v1/chat/completions")
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", env::var("GROQ_API_KEY").expect("GROQ_API_KEY must be set")))
        .json(&request_body)
        .send()
        .await;
    
    match response {
        Ok(response) => {
            let response_json = response.json::<serde_json::Value>().await.unwrap();
            let output_content = response_json["choices"][0]["message"]["content"].as_str().unwrap_or("");
            let output_tokens = count_tokens_str(output_content);

            update_token_data(client_ip, input_tokens, output_tokens).expect("Failed to update token data");

            
            let response_body = serde_json::json!({
                "response": output_content,
                "conversation_id": chat_request.conversation_id,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens
            });
            (Status::Ok, (ContentType::JSON, response_body.to_string()))
        }
        Ok(response) => (Status::BadGateway, (ContentType::JSON, format!("{{\"response\": \"Bad Gateway: {}\"}}", response.status()).to_string())),
        Err(_) => (Status::InternalServerError, (ContentType::JSON, "{\"response\": \"Internal Server Error\"}".to_string()))
    }
}

fn check_rate_limit(ip: IpAddr) -> bool {
    let mut counts = REQUEST_COUNTS.lock().unwrap();
    let (count, timestamp) = counts.entry(ip).or_insert((0, Instant::now()));

    if timestamp.elapsed() > RATE_LIMIT_TIME_WINDOW {
        *count = 0;
        *timestamp = Instant::now();
    }

    if *count < RATE_LIMIT {
        *count += 1;
        true
    } else {
        false
    }
}

fn count_tokens(messages: &Vec<Message>) -> usize {
    messages.iter().map(|m| count_tokens_str(&m.content)).sum()
}

fn count_tokens_str(content: &str) -> usize {
    content.split_whitespace().count()
}

fn update_token_data(client_ip: Option<IpAddr>, input_tokens: usize, output_tokens: usize) -> io::Result<()> {
    let file_path = env::var("TOKEN_DATA_FILE").expect("TOKEN_DATA_FILE must be set");

    check_or_create_file(&file_path)?;

    let file = OpenOptions::new().read(true).write(true).create(true).open(&file_path)?;
    let reader = BufReader::new(file);
    let mut token_data: TokenData = serde_json::from_reader(reader).unwrap_or_default();

    token_data.total_input_tokens += input_tokens;
    token_data.total_output_tokens += output_tokens;

    if let Some(ip) = client_ip {
        let ip_str = ip.to_string();
        let entry = token_data.ip_token_counts.entry(ip_str).or_insert((0, 0));
        entry.0 += input_tokens;
        entry.1 += output_tokens;
    }

    let file = OpenOptions::new().write(true).truncate(true).open(&file_path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer(writer, &token_data)?;

    Ok(())
}

fn check_or_create_file(file_path: &String) -> io::Result<()> {
    if !std::path::Path::new(file_path).exists() {
        let file = OpenOptions::new().write(true).create(true).open(file_path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer(writer, &TokenData::default())?;
    }
    Ok(())
}

#[launch]
fn rocket() -> _ {
    dotenv().ok();
    rocket::build().mount("/v1", routes![index, message, models])
        .register("/", catchers![not_found])
}