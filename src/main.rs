#[macro_use] extern crate rocket;

use rocket::http::{Status, ContentType};
use rocket::serde::json::Json;
use serde::{Deserialize, Serialize};
use reqwest::Client;
use dotenv::dotenv;
use std::io::{self, BufReader, BufWriter};
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::net::IpAddr;
use std::env;

#[derive(Deserialize, Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatRequest {    
    conversation_id: String,
    model: String,
    messages: Vec<Message>,
}

#[derive(Deserialize, Serialize, Default)]
struct TokenData {
    total_input_tokens: usize,
    total_output_tokens: usize,
    ip_token_counts: HashMap<String, (usize, usize)>,
}

#[get("/")]
fn index() -> (Status, (ContentType, &'static str))  {
    (Status::Ok, (ContentType::JSON, "{\"message\": \"Hello from luuma!\"}"))
}

#[post("/message", data = "<chat_request>")]
async fn message(chat_request: Json<ChatRequest>, client_ip: Option<IpAddr>) -> (Status, (ContentType, String)) {
    let groq_api_key = env::var("GROQ_API_KEY").expect("GROQ_API_KEY must be set");

    let input_tokens = count_tokens(&chat_request.messages);
    
    let request_body = serde_json::json!({
        "messages": chat_request.messages,
        "model": chat_request.model,
        "stream": false
    });

    let client = Client::new();
    let response = client.post("https://api.groq.com/openai/v1/chat/completions")
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", groq_api_key))
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
        Err(_) => (Status::InternalServerError, (ContentType::JSON, "{\"response\": \"Internal Server Error\"}".to_string()))
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
    rocket::build().mount("/", routes![index])
        .mount("/", routes![message])
}