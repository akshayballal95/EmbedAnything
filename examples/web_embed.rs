use candle_core::Tensor;
use embed_anything::{
    config::{EmbedConfig, JinaConfig},
    embed_query, embed_webpage,
};

fn main() {
    let start_time = std::time::Instant::now();
    let url = "https://www.scrapingbee.com/blog/web-scraping-rust/".to_string();
    let embeder = "Jina".to_string();

    let jina_config = JinaConfig {
        model_id: Some("jinaai/jina-embeddings-v2-base-en".to_string()),
        revision: None,
        chunk_size: Some(1000),
    };

    let embed_config = EmbedConfig {
        jina: Some(jina_config),
        ..Default::default()
    };

    let embed_data = embed_webpage(url, &embeder, Some(&embed_config), None)
        .unwrap()
        .unwrap();
    let embeddings: Vec<f32> = embed_data
        .iter()
        .flat_map(|data| data.embedding.iter().cloned())
        .collect();

    let num_embeddings = embed_data.len();
    let embedding_dim = embed_data[0].embedding.len();

    let embeddings_tensor = Tensor::from_vec(
        embeddings,
        (num_embeddings, embedding_dim),
        &candle_core::Device::Cpu,
    )
    .unwrap();

    let query = vec!["Rust for web scraping".to_string()];
    let query_embedding = embed_query(query, &embeder, Some(&embed_config))
        .unwrap()
        .iter()
        .flat_map(|data| data.embedding.iter().cloned())
        .collect::<Vec<f32>>();

    let query_embedding_tensor = Tensor::from_vec(
        query_embedding,
        (1, embedding_dim),
        &candle_core::Device::Cpu,
    )
    .unwrap();

    let similarities = embeddings_tensor
        .matmul(&query_embedding_tensor.transpose(0, 1).unwrap())
        .unwrap()
        .detach()
        .squeeze(1)
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let max_similarity_index = similarities
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    let data = &embed_data[max_similarity_index].metadata;

    println!("{:?}", data);
    println!("Time taken: {:?}", start_time.elapsed());
}
