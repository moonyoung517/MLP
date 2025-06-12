class IDGenerator:
    def __init__(self, start=1):
        self.current_id = start
    def get_next(self):
        current = self.current_id
        self.current_id += 1
        return current
