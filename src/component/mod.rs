use std::{collections::LinkedList, time::Duration};

use crate::sys::llm::{Content, LlamaCtx, LlamaModelChatStream, SimpleOption};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Layout},
    widgets::{Block, Paragraph, Tabs},
    Frame, Terminal,
};

pub mod chat;
pub mod lab;

#[derive(Debug)]
pub enum Input {
    Event(Event),
    Token(Option<String>),
}

#[derive(Debug)]
pub enum Output {
    Exit,
    Chat,
    Normal,
}

pub struct App {
    pub select_tabs: usize,
    pub exit_n: u8,
    pub chat: chat::ChatComponent,
    pub lab: lab::Lab,
    pub prompts_path: String,
}

impl App {
    pub fn new(prompts_path: String) -> Self {
        Self {
            chat: chat::ChatComponent::new(),
            lab: lab::Lab {
                prompts_path: prompts_path.clone(),
                messages: chat::MessagesComponent::new(),
            },
            select_tabs: 0,
            exit_n: 0,
            prompts_path,
        }
    }

    pub fn render(&mut self, contents: &LinkedList<Content>, f: &mut Frame) {
        let vertical = Layout::vertical([
            Constraint::Length(3),
            Constraint::Min(3),
            Constraint::Length(1),
            Constraint::Length(1),
        ]);

        let [tabs_area, main_area, help_area, event_area] = vertical.areas(f.size());

        let tabs = Tabs::new(vec!["Chat", "Lab"])
            .select(self.select_tabs)
            .padding("[", "]")
            .block(Block::bordered());

        f.render_widget(tabs, tabs_area);
        match self.select_tabs {
            0 => self.chat.render(contents, f, main_area),
            _ => self.lab.render(contents, f, main_area),
        }

        let help_message = Paragraph::new(format!("help: [Ctrl+R rewrite] [Esc+Esc quit]"));
        f.render_widget(help_message, help_area);

        let help_message = Paragraph::new(format!("{}", self.chat.event));
        f.render_widget(help_message, event_area);
    }

    pub fn handler_input(
        &mut self,
        input: Input,
        contents: &mut LinkedList<Content>,
        stream: &mut Option<LlamaModelChatStream<LlamaCtx>>,
    ) -> anyhow::Result<Output> {
        let last_exit_n = self.exit_n;
        if matches!(input, Input::Event(..)) {
            self.exit_n = 0;
        }
        match input {
            Input::Token(None) => {
                stream.take();
                Ok(Output::Normal)
            }
            Input::Token(Some(token)) => {
                if let Some(content) = contents.back_mut() {
                    content.message.push_str(&token);
                    let is_stop = if let Some(s) = stream {
                        s.is_stop(&mut content.message)
                    } else {
                        true
                    };
                    if is_stop {
                        stream.take();
                    }
                }
                Ok(Output::Normal)
            }
            Input::Event(Event::Key(event))
                if event.code == KeyCode::Char('c')
                    && event.modifiers.contains(KeyModifiers::CONTROL) =>
            {
                stream.take();
                Ok(Output::Normal)
            }
            Input::Event(Event::Key(event)) if event.code == KeyCode::Tab => {
                self.select_tabs = (self.select_tabs + 1) % 2;
                Ok(Output::Normal)
            }
            Input::Event(Event::Key(input)) if input.code == KeyCode::Esc => {
                self.exit_n = 1;
                if last_exit_n != 0 {
                    Ok(Output::Exit)
                } else {
                    Ok(Output::Normal)
                }
            }
            input => match self.select_tabs {
                0 => Ok(self.chat.handler_input(input, contents)),
                _ => self.lab.handler_input(input, contents),
            },
        }
    }

    pub fn get_input(stream: &mut Option<LlamaModelChatStream<LlamaCtx>>) -> anyhow::Result<Input> {
        let input = if let Some(stream_) = stream {
            // interrupt
            let input = if event::poll(Duration::from_secs(0))? {
                match event::read()? {
                    Event::Key(input)
                        if input.code == KeyCode::Char('c')
                            && input.modifiers.contains(KeyModifiers::CONTROL) =>
                    {
                        Some(Input::Event(Event::Key(input)))
                    }
                    Event::Mouse(input) => Some(Input::Event(Event::Mouse(input))),
                    _ => None,
                }
            } else {
                None
            };

            match input {
                Some(input) => input,
                None => Input::Token(stream_.next_token()?),
            }
        } else {
            Input::Event(event::read()?)
        };

        Ok(input)
    }

    pub fn run_loop(mut self, llama: &mut LlamaCtx) -> anyhow::Result<()> {
        // setup terminal
        enable_raw_mode()?;
        let mut stdout = std::io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        let mut prompts = crate::loader_prompt(&self.prompts_path)?;

        let mut main_loop = || -> anyhow::Result<()> {
            let mut stream: Option<LlamaModelChatStream<_>> = None;

            terminal.draw(|f| self.render(&prompts, f))?;

            loop {
                let input = Self::get_input(&mut stream)?;

                let output = self.handler_input(input, &mut prompts, &mut stream)?;
                terminal.draw(|f| self.render(&prompts, f))?;

                match output {
                    Output::Exit => break,
                    Output::Chat => {
                        // let option = simple_llama::SimpleOption::Temp(0.9);
                        // let option = simple_llama::SimpleOption::TopP(1.0, 20);
                        let option = SimpleOption::MirostatV2(4.0, 0.25);
                        // let option = simple_llama::SimpleOption::MirostatV2(2.0, 0.25);
                        stream = Some(llama.chat(&prompts, option).unwrap())
                    }
                    Output::Normal => {}
                }
            }
            Ok(())
        };

        let r = main_loop();

        // restore terminal
        disable_raw_mode()?;
        execute!(
            terminal.backend_mut(),
            LeaveAlternateScreen,
            DisableMouseCapture
        )?;
        terminal.show_cursor()?;
        r
    }
}
