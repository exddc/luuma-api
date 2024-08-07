use super::rocket;
use rocket::local::blocking::Client;
use rocket::http::Status;
use rocket::serde::json::Value;

const BASE_URL: &str = "/v1";

#[test]
fn index() {
    let client = Client::tracked(rocket()).unwrap();

    let response = client.get(
        format!("{}/", BASE_URL)
    ).dispatch();
    assert_eq!(response.status(), Status::Ok);
    
    let body_str = response.into_string().unwrap();
    let json: Value = serde_json::from_str(&body_str).unwrap();
    assert_eq!(json["message"], "Hello from luuma!");
}

#[test]
fn not_found() {
    let client = Client::tracked(rocket()).unwrap();

    let invalid_paths = vec![
        format!("{}/not-found", BASE_URL),
        format!("{}/not-found/", BASE_URL),
        format!("{}/1", BASE_URL),
        "/".to_string(),
        "/test".to_string(),
        "/test/".to_string(),
    ];

    for path in invalid_paths {
        let response = client.get(path).dispatch();
        assert_eq!(response.status(), Status::NotFound);
        
        let body_str = response.into_string().unwrap();
        let json: Value = serde_json::from_str(&body_str).unwrap();
        assert_eq!(json["message"], "The requested resource was not found.");
    }
}