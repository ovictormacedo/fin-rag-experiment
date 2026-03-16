from models.memory import Memory

class MemoryService:
    def __init__(self):
        self.memory = []
    
    def add_memory(self, query, context, answer):
        self.memory.append(Memory(query, context, answer))
    
    def get_latest_memory(self):
        return self.memory[-1] if self.memory else ""