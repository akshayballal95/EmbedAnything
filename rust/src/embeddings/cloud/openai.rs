use std::collections::HashMap;

use reqwest::Client;
use serde::Deserialize;
use serde_json::json;

use crate::embeddings::embed::{EmbedData, TextEmbed};

#[derive(Deserialize, Debug, Default)]
pub struct OpenAIEmbedResponse {
    pub data: Vec<EmbedData>,
    pub usage: HashMap<String, usize>,
}

/// Represents an OpenAIEmbeder struct that contains the URL and API key for making requests to the OpenAI API.
#[derive(Debug)]
pub struct OpenAIEmbeder {
    url: String,
    model: String,
    api_key: String,
    runtime: tokio::runtime::Runtime,
    client: Client,
}

impl Default for OpenAIEmbeder {
    fn default() -> Self {
        Self::new("text-embedding-3-small".to_string(), None)
    }
}

impl TextEmbed for OpenAIEmbeder {
    fn embed(
        &self,
        text_batch: &[String],
        _batch_size: Option<usize>,
    ) -> Result<Vec<Vec<f32>>, anyhow::Error> {
        self.embed(text_batch)
    }
}

impl OpenAIEmbeder {
    pub fn new(model: String, api_key: Option<String>) -> Self {
        let api_key =
            api_key.unwrap_or_else(|| std::env::var("OPENAI_API_KEY").expect("API Key not set"));

        Self {
            model,
            url: "https://api.openai.com/v1/embeddings".to_string(),
            api_key,
            runtime: tokio::runtime::Builder::new_current_thread()
                .enable_io()
                .build()
                .unwrap(),
            client: Client::new(),
        }
    }

    pub fn embed(&self, text_batch: &[String]) -> Result<Vec<Vec<f32>>, anyhow::Error> {
        let data = self.runtime.block_on(async move {
            let response = self
                .client
                .post(&self.url)
                .header("Content-Type", "application/json")
                .header("Authorization", format!("Bearer {}", self.api_key))
                .json(&json!({
                    "input": text_batch,
                    "model": self.model,
                }))
                .send()
                .await
                .unwrap();

            let data = response.json::<OpenAIEmbedResponse>().await.unwrap();
            println!("{:?}", data.usage);
            data
        });

        let encodings = data
            .data
            .iter()
            .map(|data| data.embedding.clone())
            .collect::<Vec<_>>();

        Ok(encodings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_openai_embed() {
        let openai = OpenAIEmbeder::default();
        let text_batch = vec![
            "Once upon a time".to_string(),
            "The quick brown fox jumps over the lazy dog".to_string(),
        ];

        let embeddings = openai.embed(&text_batch).unwrap();
        assert_eq!(embeddings.len(), 2);
    }
}
