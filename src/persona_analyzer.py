import spacy
from typing import List

class PersonaAnalyzer:
    def __init__(self, persona: str, job: str):
        self.nlp = spacy.load("en_core_web_sm")
        self.persona = persona.lower()
        self.job = job.lower()
        self.keywords = self._extract_keywords()
        
    def _extract_keywords(self) -> List[str]:
        """Extract relevant keywords from persona and job description"""
        doc = self.nlp(f"{self.persona} {self.job}")
        keywords = []
        
        # Extract nouns and verbs
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN", "VERB"] and not token.is_stop:
                keywords.append(token.lemma_.lower())
        
        # Add domain-specific terms based on persona
        if "researcher" in self.persona:
            keywords.extend(["methodology", "dataset", "result", "analysis", "experiment"])
        elif "analyst" in self.persona:
            keywords.extend(["trend", "investment", "strategy", "market", "financial"])
        elif "student" in self.persona:
            keywords.extend(["concept", "exam", "study", "key", "important"])
        elif "travel" in self.persona:
            keywords.extend(["itinerary", "accommodation", "transport", "attraction"])
        elif "hr" in self.persona:
            keywords.extend(["form", "onboarding", "compliance", "document"])
        elif "food" in self.persona:
            keywords.extend(["recipe", "ingredient", "preparation", "menu"])
            
        return list(set(keywords))
    
    def score_relevance(self, text: str) -> float:
        """Score text relevance based on persona and job"""
        text = text.lower()
        doc = self.nlp(text)
        
        # Count keyword matches
        keyword_matches = sum(
            1 for token in doc 
            if token.lemma_.lower() in self.keywords and not token.is_stop
        )
        
        # Calculate density score
        word_count = max(1, len([t for t in doc if not t.is_punct]))
        density_score = keyword_matches / word_count
        
        # Boost score for higher-level headings
        if any(t in text for t in ["introduction", "summary", "conclusion"]):
            density_score *= 1.5
        elif any(t in text for t in ["method", "result", "analysis"]):
            density_score *= 1.3
            
        return min(density_score * 100, 100)  # Scale to percentage, cap at 100