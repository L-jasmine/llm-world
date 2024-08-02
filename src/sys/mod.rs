use llm::{Content, Role};

pub mod llm;

pub struct NPC {
    pub name: String,
    pub description: String,

    pub character: Vec<String>,
    pub mood: String,
    pub experience: Vec<String>,
    pub current_map: String,
    pub state: String,

    pub player_relation: String,
    pub player_character: String,
}

pub struct Map {
    pub name: String,
    pub description: String,
    pub npcs: Vec<String>,
}

pub struct World {
    pub description: String,
}

pub struct StoryGenerator {
    pub prompt: String,
}

pub struct ChatGenerator {
    pub templates: String,
}

impl NPC {
    pub fn chat_system(&self, npc: &NPC) -> Content {
        Content {
            role: Role::System,
            message: String::new(),
        }
    }
}
