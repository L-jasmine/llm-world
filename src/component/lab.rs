use std::collections::{HashMap, LinkedList};

use crossterm::event::{Event, KeyCode, KeyModifiers};
use ratatui::{layout::Rect, Frame};
use simple_llama::Content;

use super::chat::{Input, MessagesComponent, Output};

pub struct Lab {
    pub prompts_path: String,
    pub messages: MessagesComponent,
}

impl Lab {
    pub fn handler_input(
        &mut self,
        input: Input,
        contents: &mut LinkedList<Content>,
    ) -> anyhow::Result<Output> {
        match input {
            Input::Event(Event::Key(event)) if event.code == KeyCode::Enter => {
                *contents = crate::loader_prompt(&self.prompts_path)?;
                Ok(Output::Chat)
            }
            Input::Event(Event::Key(event))
                if event.code == KeyCode::Char('s')
                    && event.modifiers.contains(KeyModifiers::CONTROL) =>
            {
                let mut map = HashMap::new();
                map.insert("content", contents);
                let contents = toml::to_string_pretty(&map)
                    .map_err(|e| anyhow::anyhow!("toml::to_string_pretty err:{e}"))?;
                std::fs::write(&self.prompts_path, contents)
                    .map_err(|e| anyhow::anyhow!("save to file err:{e}"))?;
                Ok(Output::Normal)
            }
            input => {
                self.messages.handler_input(input);
                Ok(Output::Normal)
            }
        }
    }

    pub fn render(&mut self, contents: &LinkedList<Content>, f: &mut Frame, area: Rect) {
        self.messages.render(contents, f, area);
    }
}
