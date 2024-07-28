use simple_llama::{Content, Role};

use crate::{chat::im_channel::Message, llm::local_llm::Token};

pub fn echo_assistant(
    tx: crossbeam::channel::Sender<Message>,
    rx: crossbeam::channel::Receiver<Message>,
) -> std::thread::JoinHandle<anyhow::Result<()>> {
    std::thread::spawn(move || {
        while let Ok(input) = rx.recv() {
            match input {
                Message::GenerateByUser(user) => {
                    let _ = tx.send(Message::Assistant(Token::Start));
                    let _ = tx.send(Message::Assistant(Token::End(user.message)));
                }
                Message::Generate(assistant) => {
                    let _ = tx.send(Message::Assistant(Token::Start));
                    let _ = tx.send(Message::Assistant(Token::End(assistant.message)));
                }
                Message::Assistant(_) => {
                    continue;
                }
            }
        }
        Ok(())
    })
}

pub struct TerminalApp {
    pub tx: crossbeam::channel::Sender<Message>,
    pub rx: crossbeam::channel::Receiver<Message>,
}

impl TerminalApp {
    pub fn filter(message: &Message) -> Option<Message> {
        if matches!(message, Message::Assistant(..)) {
            None
        } else {
            Some(message.clone())
        }
    }

    fn listen_user_input(tx: crossbeam::channel::Sender<Message>) {
        let stdin = std::io::stdin();
        loop {
            let mut line = String::new();
            let _ = stdin.read_line(&mut line).unwrap();
            if line.starts_with("exit!") {
                break;
            }
            let _ = tx.send(Message::GenerateByUser(Content {
                role: Role::User,
                message: line,
            }));
        }
    }

    pub fn run_loop(self) -> anyhow::Result<()> {
        let (input_tx, input_rx) = crossbeam::channel::unbounded();

        // setup terminal

        // create app and run it
        std::thread::spawn(move || Self::listen_user_input(input_tx));

        loop {
            let input = crossbeam::select! {
                recv(input_rx) -> input =>{
                    if let Ok(input) = input {
                        input
                    }else{
                        break;
                    }
                }
                recv(self.rx) -> message =>{
                    if let Ok(message) = message {
                        message
                    }else{
                        break;
                    }
                }
            };

            println!("{input:?}");

            match input {
                Message::Assistant(..) => {}
                input => {
                    let _ = self.tx.send(input);
                }
            }
        }

        Ok(())
    }
}
