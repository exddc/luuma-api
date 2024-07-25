#[macro_use] extern crate rocket;
use rocket::http::{Status, ContentType};

#[get("/")]
fn index() -> (Status, (ContentType, &'static str))  {
    (Status::Ok, (ContentType::JSON, "{\"message\": \"Hello from luuma!\"}"))
}

#[launch]
fn rocket() -> _ {
    rocket::build().mount("/", routes![index])
}