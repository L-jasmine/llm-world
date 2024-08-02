use std::{
    collections::{HashMap, LinkedList},
    error::Error,
    num::NonZeroU32,
};

use anyhow::anyhow;
use clap::Parser;
use sys::llm::{Content, LlamaContextParams, LlamaCtx, LlamaModelParams, LlmModel, PromptTemplate};

mod component;
mod sys;

#[derive(Debug, clap::Parser)]
struct Args {
    #[arg(long, short, required = true)]
    project_path: String,

    /// full prompt chat
    #[arg(long)]
    debug_ui: bool,

    #[arg(long)]
    debug_llm: bool,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct Project {
    model_path: String,
    prompts: String,
    template: String,
    run: RunOptions,
    templates: HashMap<String, PromptTemplate>,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct RunOptions {
    #[serde(default)]
    ctx_size: u32,
    #[serde(default)]
    n_batch: u32,
    #[serde(default)]
    n_gpu_layers: u32,
}

impl RunOptions {
    fn fill_default_value(&mut self) {
        if self.ctx_size == 0 {
            self.ctx_size = 1024;
        }
        if self.n_batch == 0 {
            self.n_batch = 512;
        }
        if self.n_gpu_layers == 0 {
            self.n_gpu_layers = 100;
        }
    }
}

#[derive(Debug, Clone, clap::ValueEnum)]
enum Engine {
    None,
    Lua,
    Rhai,
}

fn loader_prompt(prompt_file_path: &str) -> anyhow::Result<LinkedList<Content>> {
    let prompt = std::fs::read_to_string(prompt_file_path)
        .map_err(|_| anyhow::anyhow!("prompt file `{}` not found", prompt_file_path))?;

    let mut prompt: HashMap<String, LinkedList<Content>> = toml::from_str(&prompt)?;
    let prompts = prompt
        .remove("content")
        .ok_or(anyhow!("'content' not exist!"))?;
    Ok(prompts)
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    let cli = Args::parse();
    let mut project: Project =
        toml::from_str(&std::fs::read_to_string(&cli.project_path).unwrap()).unwrap();
    project.run.fill_default_value();

    let template = project
        .templates
        .get(&project.template)
        .ok_or(anyhow::anyhow!("template not found"))?
        .clone();

    let model_params: LlamaModelParams =
        LlamaModelParams::default().with_n_gpu_layers(project.run.n_gpu_layers);

    let llm = LlmModel::new(project.model_path, model_params, template)
        .map_err(|e| anyhow::anyhow!(e))?;

    let ctx_params = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(project.run.ctx_size))
        .with_n_batch(project.run.n_batch);

    let mut ctx = LlamaCtx::new(llm, ctx_params).unwrap();

    let app = component::App::new(project.prompts.clone());

    let res = app.run_loop(&mut ctx);

    if let Err(err) = res {
        println!("{err:?}");
    }

    Ok(())
}
