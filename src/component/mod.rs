use std::{collections::LinkedList, time::Duration};

use chat::{Input, Output};
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
use simple_llama::{Content, LlamaCtx, LlamaModelChatStream};

pub mod chat;

#[derive(Debug, Clone)]
pub enum Token {
    Chunk(String),
    End(String),
}

pub struct App {
    pub chat: chat::ChatComponent,
}

impl App {
    pub fn new(prompts: LinkedList<Content>) -> Self {
        Self {
            chat: chat::ChatComponent::new(prompts),
        }
    }

    fn prompts(&self) -> &LinkedList<Content> {
        &self.chat.messages.contents
    }

    pub fn render(&mut self, f: &mut Frame) {
        let vertical = Layout::vertical([
            Constraint::Length(3),
            Constraint::Min(3),
            Constraint::Length(1),
            Constraint::Length(1),
        ]);

        let [tabs_area, main_area, help_area, event_area] = vertical.areas(f.size());

        let tabs = Tabs::new(vec!["Chat", "Setting"])
            .select(0)
            .padding("[", "]")
            .block(Block::bordered());

        f.render_widget(tabs, tabs_area);
        self.chat.render(f, main_area);

        let help_message = Paragraph::new(format!("help: [Ctrl+R rewrite] [Esc+Esc quit]"));
        f.render_widget(help_message, help_area);

        let help_message = Paragraph::new(format!("{}", self.chat.event));
        f.render_widget(help_message, event_area);
    }

    pub fn run_loop(mut self, llama: &mut LlamaCtx) -> anyhow::Result<()> {
        // setup terminal
        enable_raw_mode()?;
        let mut stdout = std::io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        let mut main_loop = || -> anyhow::Result<()> {
            let mut output = Output::Normal;
            let mut stream: Option<LlamaModelChatStream<_>> = None;
            let mut start_content = None;

            loop {
                terminal.draw(|f| self.render(f))?;

                match output {
                    Output::Exit => break,
                    Output::Chat => {
                        if let Some(c) = self.prompts().back() {
                            start_content = Some(c.message.clone());
                        }
                        stream = Some(
                            llama
                                .chat(self.prompts(), simple_llama::SimpleOption::Temp(0.9))
                                .unwrap(),
                        )
                    }
                    Output::Normal => {}
                }

                let input = if let Some(mut stream_) = stream.take() {
                    // interrupt
                    if event::poll(Duration::from_secs(0))? {
                        if let Event::Key(s) = event::read()? {
                            if s.code == KeyCode::Char('c')
                                && s.modifiers.contains(KeyModifiers::CONTROL)
                            {
                                continue;
                            }
                        }
                    };

                    if let Some(token) = stream_.next() {
                        stream = Some(stream_);
                        Input::Message(Token::Chunk(token))
                    } else {
                        let end: String = stream_.into();
                        let start_content = start_content.take().unwrap_or_default();
                        Input::Message(Token::End(start_content + &end))
                    }
                } else {
                    Input::Event(event::read()?)
                };

                output = self.chat.handler_input(&mut terminal, input);
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
