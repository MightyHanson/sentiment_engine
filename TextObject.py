# Text Object Class
from typing import Optional
class TextObject:
    def __init__(self, source: str, text_content: str, metadata: Optional[dict] = None):
        self.source = source
        self.text_content = text_content
        self.metadata = metadata or {}