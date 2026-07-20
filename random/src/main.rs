use random::{respond, BotRequest};
use serde::Serialize;
use std::io::{self, BufRead};

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

fn main() {
    for line in io::stdin().lock().lines() {
        let output = match line {
            Ok(line) => serde_json::from_str::<BotRequest>(&line)
                .map_err(|error| error.to_string())
                .and_then(respond)
                .and_then(|response| serde_json::to_string(&response).map_err(|e| e.to_string())),
            Err(error) => Err(error.to_string()),
        };

        match output {
            Ok(json) => println!("{json}"),
            Err(error) => println!(
                "{}",
                serde_json::to_string(&ErrorResponse { error }).unwrap()
            ),
        }
    }
}
