import json
import os
from datetime import datetime
from ollama import Client

class SimpleLLMMemory:
    def __init__(self, ollama_url="http://localhost:11434", model="mistral:latest", memory_file="memory.json"):
        self.client = Client(host=ollama_url)
        self.model = model
        self.memory_file = memory_file
        
        # Volatile memory for current session
        self.conversation_history = []  # List of {"role": "user/assistant", "content": "...", "timestamp": "..."}
        
        # Load persistent memory from file
        self.persistent_memory = self.load_memory()
        
        # System prompt
        self.system_prompt = """You are a helpful AI assistant with access to conversation context and learned facts.

CONTEXT PROVIDED:
- Recent conversation history (last 3 exchanges)
- Important facts and relationships from previous conversations

Use this context to provide more personalized and contextually aware responses. Reference previous conversations when relevant, but don't always mention that you remember things unless it adds value to the response.

Be natural and conversational while leveraging the provided context."""

        # Fact extraction prompt
        self.fact_extraction_prompt = """You are a fact extractor. Extract important facts and relationships from the user's message.

User message: "{conversation}"

Extract facts about:
- Personal information (name, age, location, job, family)
- Interests, hobbies, preferences
- Skills, background, goals
- Relationships between people, concepts, or preferences

Return ONLY a valid JSON object like this:
{{
    "facts": ["User likes playing football", "User's name is Shaim"],
    "relationships": ["John is user's father", "Ziya is user's sister"]
}}

Important: Return ONLY the JSON object, no other text."""

    def load_memory(self):
        """Load persistent memory from JSON file"""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                print(f"Error loading {self.memory_file}, creating new memory")
        
        # Default structure
        default_memory = {
            "facts": [],
            "relationships": [],
            "summaries": [],
            "last_updated": datetime.now().isoformat()
        }
        
        # Create the file immediately
        self.save_memory_data(default_memory)
        return default_memory

    def save_memory_data(self, data):
        """Save memory data to JSON file"""
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Memory saved to {self.memory_file}")
        except Exception as e:
            print(f"Error saving memory: {e}")

    def save_memory(self):
        """Save persistent memory to JSON file"""
        self.persistent_memory["last_updated"] = datetime.now().isoformat()
        self.save_memory_data(self.persistent_memory)

    def add_to_conversation(self, role, content):
        """Add a message to conversation history"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.conversation_history.append(message)

    def get_recent_context(self, num_exchanges=3):
        """Get the last N conversation exchanges (user + assistant pairs)"""
        if not self.conversation_history:
            return []
        
        # Get last num_exchanges * 2 messages (user + assistant pairs)
        recent_messages = self.conversation_history[-(num_exchanges * 2):]
        return recent_messages

    def extract_facts_and_relationships(self):
        """Use LLM to extract facts and relationships from recent conversation"""
        if len(self.conversation_history) < 2:
            return
        
        # Get recent conversation for analysis
        recent_conv = self.get_recent_context(2)  # Last 2 exchanges
        if not recent_conv:
            return
        
        # Format conversation for analysis
        conv_text = []
        for msg in recent_conv:
            role_label = "User" if msg["role"] == "user" else "Assistant"
            conv_text.append(f"{role_label}: {msg['content']}")
        
        conversation_str = "\n".join(conv_text)
        
        try:
            # Call LLM to extract facts
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "user", "content": self.fact_extraction_prompt.format(conversation=conversation_str)}
                ]
            )
            
            extracted_text = response['message']['content'].strip()
            
            # Try to parse JSON response
            try:
                extracted_data = json.loads(extracted_text)
                
                # Add new facts
                if "facts" in extracted_data and extracted_data["facts"]:
                    for fact in extracted_data["facts"]:
                        if fact and fact not in self.persistent_memory["facts"]:
                            self.persistent_memory["facts"].append(fact)
                            print(f"💡 Learned fact: {fact}")
                
                # Add new relationships
                if "relationships" in extracted_data and extracted_data["relationships"]:
                    for rel in extracted_data["relationships"]:
                        if rel and rel not in self.persistent_memory["relationships"]:
                            self.persistent_memory["relationships"].append(rel)
                            print(f"🔗 Learned relationship: {rel}")
                
                # Save if we learned something new
                if (extracted_data.get("facts") or extracted_data.get("relationships")):
                    self.save_memory()
                    
            except json.JSONDecodeError:
                print(f"Could not parse extracted facts: {extracted_text}")
                
        except Exception as e:
            print(f"Error extracting facts: {e}")

    def build_context_prompt(self, user_input):
        """Build the full prompt with context"""
        context_parts = []
        
        # Add recent conversation history
        recent_context = self.get_recent_context()
        if recent_context:
            context_parts.append("RECENT CONVERSATION:")
            for msg in recent_context:
                role_label = "User" if msg["role"] == "user" else "Assistant"
                context_parts.append(f"{role_label}: {msg['content']}")
            context_parts.append("")
        
        # Add facts
        if self.persistent_memory["facts"]:
            context_parts.append("KNOWN FACTS:")
            for fact in self.persistent_memory["facts"][-8:]:  # Last 8 facts
                context_parts.append(f"- {fact}")
            context_parts.append("")
        
        # Add relationships
        if self.persistent_memory["relationships"]:
            context_parts.append("KNOWN RELATIONSHIPS:")
            for rel in self.persistent_memory["relationships"][-5:]:  # Last 5 relationships
                context_parts.append(f"- {rel}")
            context_parts.append("")
        
        # Add summaries
        if self.persistent_memory["summaries"]:
            context_parts.append("CONVERSATION SUMMARIES:")
            for summary in self.persistent_memory["summaries"][-3:]:  # Last 3 summaries
                if isinstance(summary, dict):
                    context_parts.append(f"- {summary['summary']}")
                else:
                    context_parts.append(f"- {summary}")
            context_parts.append("")
        
        # Add current user input
        context_parts.append(f"Current User Input: {user_input}")
        
        return "\n".join(context_parts)

    def add_fact(self, fact):
        """Manually add an important fact to persistent memory"""
        if fact and fact not in self.persistent_memory["facts"]:
            self.persistent_memory["facts"].append(fact)
            self.save_memory()
            print(f"Added fact: {fact}")

    def add_relationship(self, relationship):
        """Manually add a relationship to persistent memory"""
        if relationship and relationship not in self.persistent_memory["relationships"]:
            self.persistent_memory["relationships"].append(relationship)
            self.save_memory()
            print(f"Added relationship: {relationship}")

    def add_summary(self, summary):
        """Add a conversation summary to persistent memory"""
        if summary:
            self.persistent_memory["summaries"].append({
                "summary": summary,
                "date": datetime.now().isoformat()
            })
            # Keep only last 10 summaries
            if len(self.persistent_memory["summaries"]) > 10:
                self.persistent_memory["summaries"] = self.persistent_memory["summaries"][-10:]
            self.save_memory()

    def extract_from_user_input(self, user_input):
        """Extract facts and relationships from user input before responding"""
        if not user_input.strip():
            return
        
        print(f"🔍 Analyzing: {user_input}")  # Debug print
        
        try:
            # Call LLM to extract facts from user input
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "user", "content": self.fact_extraction_prompt.format(conversation=user_input)}
                ]
            )
            
            extracted_text = response['message']['content'].strip()
            print(f"🔍 LLM Response: {extracted_text}")  # Debug print
            
            # Try to parse JSON response
            try:
                # Clean the response - sometimes LLM adds extra text
                if extracted_text.startswith('```json'):
                    extracted_text = extracted_text.replace('```json', '').replace('```', '').strip()
                elif extracted_text.startswith('```'):
                    extracted_text = extracted_text.replace('```', '').strip()
                
                # Find JSON object in the response
                start_idx = extracted_text.find('{')
                end_idx = extracted_text.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = extracted_text[start_idx:end_idx]
                    extracted_data = json.loads(json_str)
                else:
                    extracted_data = json.loads(extracted_text)
                
                # Add new facts
                if "facts" in extracted_data and extracted_data["facts"]:
                    for fact in extracted_data["facts"]:
                        if fact and fact not in self.persistent_memory["facts"]:
                            self.persistent_memory["facts"].append(fact)
                            print(f"💡 Learned: {fact}")
                
                # Add new relationships
                if "relationships" in extracted_data and extracted_data["relationships"]:
                    for rel in extracted_data["relationships"]:
                        if rel and rel not in self.persistent_memory["relationships"]:
                            self.persistent_memory["relationships"].append(rel)
                            print(f"🔗 Connected: {rel}")
                
                # Save if we learned something new
                if (extracted_data.get("facts") or extracted_data.get("relationships")):
                    self.save_memory()
                    
            except json.JSONDecodeError as e:
                print(f"❌ JSON Parse Error: {e}")  # Debug print
                print(f"Raw response: {extracted_text}")
                
        except Exception as e:
            print(f"❌ Extraction Error: {e}")  # Debug print

    def chat(self, user_input):
        """Main chat function with memory"""
        # Extract facts and relationships from user input first
        self.extract_from_user_input(user_input)
        
        # Add user input to conversation history
        self.add_to_conversation("user", user_input)
        
        # Build context prompt
        full_prompt = self.build_context_prompt(user_input)
        
        try:
            # Call the LLM
            response = self.client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": full_prompt}
                ]
            )
            
            assistant_response = response['message']['content']
            
            # Add assistant response to conversation history
            self.add_to_conversation("assistant", assistant_response)
            
            # Keep conversation history manageable (last 20 messages)
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]
            
            return assistant_response
            
        except Exception as e:
            return f"Error communicating with LLM: {str(e)}"

    def clear_session_memory(self):
        """Clear current session conversation history"""
        self.conversation_history = []

    def show_memory_stats(self):
        """Display current memory statistics"""
        print(f"\n📊 Memory Statistics:")
        print(f"Current session messages: {len(self.conversation_history)}")
        print(f"Stored facts: {len(self.persistent_memory['facts'])}")
        print(f"Stored relationships: {len(self.persistent_memory['relationships'])}")
        print(f"Stored summaries: {len(self.persistent_memory['summaries'])}")
        print(f"Last updated: {self.persistent_memory['last_updated']}")
        
        if self.persistent_memory['facts']:
            print(f"\n📝 Recent Facts:")
            for fact in self.persistent_memory['facts'][-3:]:
                print(f"  - {fact}")
        
        if self.persistent_memory['relationships']:
            print(f"\n🔗 Recent Relationships:")
            for rel in self.persistent_memory['relationships'][-3:]:
                print(f"  - {rel}")

    def show_memory_file_location(self):
        """Show the location of the memory file"""
        abs_path = os.path.abspath(self.memory_file)
        print(f"💾 Memory file location: {abs_path}")
        if os.path.exists(self.memory_file):
            print(f"✅ File exists and is {os.path.getsize(self.memory_file)} bytes")
        else:
            print("❌ File does not exist yet")


def main():
    """Example usage"""
    # Initialize the memory system
    llm_memory = SimpleLLMMemory()
    
    print("🧠 Simple LLM with Auto-Learning Memory")
    print("Just chat naturally - I'll automatically learn about you!")
    print("Commands: /stats, /clear, /file, /quit")
    print("-" * 50)
    
    # Show file location
    llm_memory.show_memory_file_location()
    
    while True:
        user_input = input("\n🗣️  You: ").strip()
        
        if user_input.lower() == '/quit':
            break
        elif user_input.lower() == '/clear':
            llm_memory.clear_session_memory()
            print("🧹 Session memory cleared!")
            continue
        elif user_input.lower() == '/stats':
            llm_memory.show_memory_stats()
            continue
        elif user_input.lower() == '/file':
            llm_memory.show_memory_file_location()
            continue
        elif not user_input:
            continue
        
        # Get response from LLM with automatic memory learning
        response = llm_memory.chat(user_input)
        print(f"\n🤖 Assistant: {response}")


if __name__ == "__main__":
    main()