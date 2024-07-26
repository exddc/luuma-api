#[macro_use] extern crate rocket;

use rocket::http::{Status, ContentType};
use rocket::serde::json::Json;
use serde::{Deserialize, Serialize};
use reqwest::Client;
use dotenv::dotenv;
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

#[get("/")]
fn index() -> (Status, (ContentType, &'static str))  {
    (Status::Ok, (ContentType::JSON, "{\"message\": \"Hello from luuma!\"}"))
}

#[post("/message", data = "<chat_request>")]
async fn message(chat_request: Json<ChatRequest>) -> (Status, (ContentType, String)) {
    let groq_api_key = env::var("GROQ_API_KEY").expect("GROQ_API_KEY must be set");
    
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
            let response_body = serde_json::json!({
                "response": response.json::<serde_json::Value>().await.unwrap()["choices"][0]["message"]["content"]
            });
            (Status::Ok, (ContentType::JSON, response_body.to_string()))
        }
        Err(_) => (Status::InternalServerError, (ContentType::JSON, "{\"response\": \"Internal Server Error\"}".to_string()))
    }
}

#[launch]
fn rocket() -> _ {
    dotenv().ok();
    rocket::build().mount("/", routes![index])
        .mount("/", routes![message])
}