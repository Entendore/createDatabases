import math
import random
import json
import sqlite3
import pronouncing
from datetime import datetime
from collections import defaultdict, Counter
from functools import lru_cache

# ==========================================
# 1. SQLite Database Management & Export
# ==========================================
class PoetryDatabase:
    def __init__(self, db_name="unsupervised_poetry.db"):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS poems (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT,
                form TEXT,
                poem_text TEXT,
                score REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()

    def save_poem(self, prompt, form, poem_text, score):
        self.cursor.execute('''
            INSERT INTO poems (prompt, form, poem_text, score)
            VALUES (?, ?, ?, ?)
        ''', (prompt, form, poem_text, score))
        self.conn.commit()

    def get_count(self):
        self.cursor.execute("SELECT COUNT(*) FROM poems")
        return self.cursor.fetchone()[0]

    def export_to_jsonl(self, filename="poetry_dataset.jsonl"):
        """Exports the database to a JSONL file for unsupervised ML training."""
        self.cursor.execute("SELECT poem_text, form, score FROM poems")
        rows = self.cursor.fetchall()
        
        with open(filename, 'w', encoding='utf-8') as f:
            for row in rows:
                # Standard JSONL format for LLM training (e.g., text, metadata)
                data = {
                    "text": row[0],
                    "form": row[1],
                    "score": row[2]
                }
                f.write(json.dumps(data) + "\n")
        print(f"\n[+] Exported {len(rows)} poems to {filename} for ML training.")

# ==========================================
# 2. Poetic Forms, Semantics & Sensory Maps
# ==========================================
POEM_FORMS = {
    "Couplet": {
        "rhyme_scheme": ['A', 'A'], "line_syllables": [6, 6],
        "line_meters": ["010101", "010101"],
        "weights": {"rhyme": 1.5, "meter": 1.5, "sensory": 1.0, "narrative": 1.0, "macro": 2.0}
    },
    "Haiku": {
        "rhyme_scheme": [None, None, None], "line_syllables": [5, 7, 5],
        "line_meters": [None, None, None],
        "weights": {"rhyme": 0.0, "meter": 0.0, "sensory": 2.5, "narrative": 2.0, "macro": 0.0} 
    },
    "Limerick": {
        "rhyme_scheme": ['A', 'A', 'B', 'B', 'A'], "line_syllables": [8, 8, 5, 5, 8],
        "line_meters": ["00100100", "00100100", "00100", "00100", "00100100"],
        "weights": {"rhyme": 2.0, "meter": 2.0, "sensory": 0.5, "narrative": 1.5, "macro": 1.5}
    }
}

SEMANTIC_MAP = {
    'night': ['DARKNESS', 'TIME', 'CELESTIAL'], 'dark': ['DARKNESS'], 'deep': ['DARKNESS', 'DEPTH'],
    'moon': ['CELESTIAL', 'LIGHT'], 'stars': ['CELESTIAL', 'LIGHT'], 'sky': ['CELESTIAL', 'NATURE'],
    'light': ['LIGHT', 'CELESTIAL'], 'bright': ['LIGHT'], 'high': ['HEIGHT', 'CELESTIAL'],
    'wind': ['NATURE', 'WEATHER'], 'forest': ['NATURE'], 'trees': ['NATURE'],
    'stone': ['NATURE', 'EARTH'], 'hill': ['NATURE', 'EARTH'], 'water': ['NATURE'],
    'fire': ['LIGHT', 'HEAT'], 'raven': ['ANIMAL', 'DARKNESS'], 'owl': ['ANIMAL', 'DARKNESS'],
    'dog': ['ANIMAL'], 'lark': ['ANIMAL', 'LIGHT'], 'falcon': ['ANIMAL', 'HEIGHT'],
    'shadows': ['DARKNESS'], 'cold': ['WEATHER'], 'sigh': ['WEATHER', 'EMOTION'],
    'moan': ['EMOTION', 'SOUND'], 'bark': ['SOUND'], 'flight': ['MOVEMENT', 'HEIGHT'],
    'wood': ['NATURE', 'EARTH'], 'home': ['STRUCTURE'], 'park': ['NATURE', 'STRUCTURE'],
    'pale': ['COLOR'], 'white': ['COLOR'], 'hero': ['PERSON'], 'man': ['PERSON']
}

SENSORY_MAP = {
    'dark': ['SIGHT'], 'bright': ['SIGHT'], 'pale': ['SIGHT'], 'white': ['SIGHT'], 
    'shadows': ['SIGHT'], 'light': ['SIGHT'], 'moon': ['SIGHT'], 'stars': ['SIGHT'],
    'sigh': ['SOUND', 'TOUCH'], 'moan': ['SOUND'], 'bark': ['SOUND'], 'wind': ['TOUCH', 'SOUND'],
    'cold': ['TOUCH'], 'fire': ['TOUCH', 'SIGHT'], 'water': ['TOUCH'], 'wood': ['TOUCH']
}

SENTIMENT_MAP = {
    'night': -1, 'dark': -1, 'deep': -1, 'shadows': -1, 'cold': -1, 'moan': -1, 
    'alone': -1, 'sigh': -1, 'pale': -1, 'raven': -1, 'owl': -1,
    'moon': 0, 'stars': 0, 'sky': 0, 'wind': 0, 'forest': 0, 'trees': 0,
    'stone': 0, 'hill': 0, 'water': 0, 'wood': 0, 'home': 0, 'park': 0, 'man': 0,
    'light': 1, 'bright': 1, 'high': 1, 'fire': 1, 'lark': 1, 'hero': 1, 
    'flight': 1, 'white': 1, 'falcon': 1
}

STOPWORDS = set("a an the and or but if then else when at by for in of on to up with is am are was were be been i you he she it we they me him her us them my your his its our their this that these those has have had do does did".split())

# ==========================================
# 3. NLP Utilities (Phonetics, Meter & Semantics)
# ==========================================
@lru_cache(maxsize=None)
def get_syllables(word):
    phones = pronouncing.phones_for_word(word.lower().strip('.,!?;:"'))
    if phones: return pronouncing.syllable_count(phones[0])
    return 1

@lru_cache(maxsize=None)
def get_rhymes(word):
    return set(pronouncing.rhymes(word.lower().strip('.,!?;:"')))

@lru_cache(maxsize=None)
def get_stress(word):
    phones = pronouncing.phones_for_word(word.lower().strip('.,!?;:"'))
    if phones: return pronouncing.stresses(phones[0])
    return ""

@lru_cache(maxsize=None)
def get_phonemes(word):
    phones = pronouncing.phones_for_word(word.lower().strip('.,!?;:"'))
    return phones[0].split() if phones else []

def do_rhyme(word1, word2):
    if word1.lower() == word2.lower(): return False
    return word2.lower() in get_rhymes(word1.lower())

def is_slant_rhyme(word1, word2):
    p1 = get_phonemes(word1)
    p2 = get_phonemes(word2)
    if not p1 or not p2: return False
    if p1[-1] == p2[-1] and not p1[-1][-1].isdigit(): return True
    if p1[-1][-1].isdigit() and p1[-1] == p2[-1]: return True
    return False

@lru_cache(maxsize=None)
def stress_distance(stress1, stress2):
    if len(stress1) < len(stress2): return stress_distance(stress2, stress1)
    if len(stress2) == 0: return len(stress1)
    previous_row = range(len(stress2) + 1)
    for i, c1 in enumerate(stress1):
        current_row = [i + 1]
        for j, c2 in enumerate(stress2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def evaluate_meter(line_stress, target_meter):
    if not target_meter: return 0.0
    if line_stress == target_meter: return 75.0
    if len(line_stress) == len(target_meter) + 1 and line_stress[-1] == '0':
        if line_stress[:-1] == target_meter: return 70.0 
    dist = stress_distance(line_stress, target_meter)
    max_dist = max(len(line_stress), len(target_meter))
    return ((max_dist - dist) / max_dist) * 75.0 if max_dist > 0 else 0

def calculate_alliteration(words):
    consonants = [p[0] for p in [get_phonemes(w) for w in words] if p and not p[0][0].isdigit()]
    if not consonants: return 0
    counts = Counter(consonants)
    max_count = max(counts.values())
    return max_count * 15.0 if max_count > 1 else 0

def get_concepts(word): return set(SEMANTIC_MAP.get(word.lower().strip('.,!?;:"'), []))
def get_sentiment(word): return SENTIMENT_MAP.get(word.lower().strip('.,!?;:"'), 0)
def get_senses(word): return set(SENSORY_MAP.get(word.lower().strip('.,!?;:"'), []))

def calculate_thematic_coherence(lines):
    score = 0.0
    for i in range(1, len(lines)):
        c1 = set().union(*[get_concepts(w) for w in lines[i-1].split()])
        c2 = set().union(*[get_concepts(w) for w in lines[i].split()])
        if c1 and c2: score += (len(c1.intersection(c2)) / len(c1.union(c2))) * 100.0 
    return score

def calculate_narrative_progression(lines):
    if len(lines) < 2: return 0
    first_half_concepts = set().union(*[get_concepts(w) for w in lines[0].split()])
    last_half_concepts = set().union(*[get_concepts(w) for w in lines[-1].split()])
    return len(last_half_concepts - first_half_concepts) * 40.0

def calculate_sentiment_arc(lines):
    if len(lines) < 2: return 0
    first_sent = sum(get_sentiment(w) for w in lines[0].split())
    last_sent = sum(get_sentiment(w) for w in lines[-1].split())
    if last_sent > first_sent: return abs(last_sent - first_sent) * 60.0
    elif last_sent < first_sent: return abs(last_sent - first_sent) * 40.0
    return 0

def calculate_lexical_density(words):
    if not words: return 0
    content_words = [w for w in words if w.lower() not in STOPWORDS]
    return (len(content_words) / len(words)) * 50.0

def calculate_sensory_engagement(lines):
    senses = set()
    for line in lines:
        for w in line.split(): senses.update(get_senses(w))
    return len(senses) * 35.0

def calculate_rhetorical_devices(lines):
    score = 0.0
    if len(lines) >= 2:
        first_words = [line.split()[0].lower() for line in lines if line.split()]
        for i in range(1, len(first_words)):
            if first_words[i] == first_words[i-1]: score += 60.0
        last_words = [line.split()[-1].strip('.,!?;:"').lower() for line in lines if line.split()]
        for i in range(1, len(last_words)):
            if last_words[i] == last_words[i-1]: score += 40.0
    return score

def calculate_enjambment_variety(lines):
    end_stopped = sum(1 for line in lines if line and line[-1] in '.,!?;:"')
    if 0 < end_stopped < len(lines): return 40.0
    return 0.0

def calculate_macro_rhythm(lines, form):
    if not form['line_meters'][0]: return 0.0
    stresses = ["".join([get_stress(w) for w in line.split()]) for line in lines]
    core_stresses = [s[:-1] if s[-1] == '0' else s for s in stresses if s]
    if len(set(core_stresses)) == 1 and len(core_stresses) > 1: return 80.0
    return 0.0

def calculate_ttr(lines):
    words = [w.lower().strip('.,!?;:"') for line in lines for w in line.split()]
    if not words: return 1.0
    return len(set(words)) / len(words)

# ==========================================
# 4. Autonomous N-Gram Model
# ==========================================
class NgramModel:
    def __init__(self, corpus, k=0.01):
        self.k = k
        self.trigrams = defaultdict(lambda: defaultdict(int))
        self.bigrams = defaultdict(lambda: defaultdict(int))
        self.vocab = set()
        
        words = corpus.lower().split()
        self.vocab.update(words)
        
        for i in range(len(words) - 2):
            self.trigrams[(words[i], words[i+1])][words[i+2]] += 1
        for i in range(len(words) - 1):
            self.bigrams[words[i]][words[i+1]] += 1

    def get_random_seed(self):
        """Selects a random strong content word to start a poem autonomously."""
        content_words = [w for w in self.vocab if w not in STOPWORDS and len(w) > 2]
        return random.choice(content_words)

    def get_proposals_with_priors(self, history, full_text, rhyme_target=None, current_syls=0, target_syls=4):
        proposals = []
        V = len(self.vocab) + 1000 
        word_counts = Counter(full_text.lower().split())
        
        def apply_penalty(word, prob):
            return prob / ((1.0 + word_counts.get(word, 0)) ** 2)
        
        if len(history) >= 2:
            key = (history[-2], history[-1])
            if key in self.trigrams:
                total_count = sum(self.trigrams[key].values())
                for w, count in sorted(self.trigrams[key].items(), key=lambda x: x[1], reverse=True)[:5]:
                    prob = (count + self.k) / (total_count + self.k * V)
                    proposals.append((w, apply_penalty(w, prob)))
        
        if len(history) >= 1:
            key = history[-1]
            if key in self.bigrams:
                total_count = sum(self.bigrams[key].values())
                for w, count in sorted(self.bigrams[key].items(), key=lambda x: x[1], reverse=True)[:5]:
                    if w not in [p[0] for p in proposals]:
                        prob = (count + self.k) / (total_count + self.k * V)
                        proposals.append((w, apply_penalty(w, prob)))
        
        if rhyme_target:
            rhymes = get_rhymes(rhyme_target)
            remaining_syls = target_syls - current_syls
            for r in rhymes:
                if r not in [p[0] for p in proposals] and get_syllables(r) == remaining_syls:
                    syl_weight = 1.0 / get_syllables(r)
                    base_prob = (self.k / V) * 100 * syl_weight
                    proposals.append((r, apply_penalty(r, base_prob)))
            
            proposals.sort(key=lambda x: x[1], reverse=True)
            forced_rhymes = [p for p in proposals if p[0] in rhymes][:2]
            other_words = [p for p in proposals if p[0] not in rhymes][:5]
            proposals = forced_rhymes + other_words
            
        return proposals[:7]

# ==========================================
# 5. MCTS Node and Dynamic State
# ==========================================
class PoemState:
    def __init__(self, text="", form_name="Couplet"):
        self.text = text
        self.form_name = form_name
        self.rhyme_map = {} 

    def get_form(self): return POEM_FORMS[self.form_name]
    def is_terminal(self): return self.text.count('\n') >= len(self.get_form()['rhyme_scheme'])

class Node:
    def __init__(self, state, parent=None, action=None, prior=0.0, c_puct=2.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = None
        self.max_children = 3 
        self.c_puct = c_puct

    def puct(self):
        if self.visits == 0: return float('inf')
        exploitation = self.value / self.visits
        exploration = self.c_puct * self.prior * (math.sqrt(self.parent.visits) / (1 + self.visits))
        return exploitation + exploration

# ==========================================
# 6. Unsupervised MCTS Engine
# ==========================================
class PoemMCTS_Unsupervised:
    def __init__(self, prompt, model, form_name="Couplet", iterations=1500, gamma=0.9):
        self.root = Node(PoemState(prompt, form_name), c_puct=2.0)
        self.model = model
        self.iterations = iterations
        self.gamma = gamma
        self.transposition_table = {}
        self.noise_injected = False

    def get_context(self, state):
        lines = state.text.split('\n')
        current_line = lines[-1].strip('.,!?;:"')
        words = current_line.split() if current_line else []
        current_syls = sum(get_syllables(w) for w in words)
        line_idx = len(lines) - 1
        return lines, words, current_syls, line_idx

    def get_valid_actions(self, node):
        lines, words, current_syls, line_idx = self.get_context(node.state)
        form = node.state.get_form()
        
        target_syls = form['line_syllables'][line_idx]
        rhyme_class = form['rhyme_scheme'][line_idx]
        rhyme_target = None
        if rhyme_class and rhyme_class in node.state.rhyme_map: rhyme_target = node.state.rhyme_map[rhyme_class]

        raw_proposals = self.model.get_proposals_with_priors(words, node.state.text, rhyme_target, current_syls, target_syls)
        valid_actions = []

        for word, prior in raw_proposals:
            next_syls = get_syllables(word)
            total_syls = current_syls + next_syls
            if total_syls > target_syls: continue

            is_corpus_word = word in self.model.vocab
            if not is_corpus_word and total_syls != target_syls: continue

            if total_syls == target_syls:
                if rhyme_target:
                    if not do_rhyme(rhyme_target, word) and not is_slant_rhyme(rhyme_target, word): continue
                if word in STOPWORDS: continue

            valid_actions.append((word, prior))
        return valid_actions

    def expand(self, node):
        if node.untried_actions is None:
            node.untried_actions = self.get_valid_actions(node)
            if node == self.root and not self.noise_injected and node.untried_actions:
                noisy_actions = [(w, p + random.random() * 0.5) for w, p in node.untried_actions]
                node.untried_actions = noisy_actions
                self.noise_injected = True

        if node.visits > 10: node.max_children = 5
        if node.visits > 25: node.max_children = 7
        
        if len(node.children) >= node.max_children: return None 
        if not node.untried_actions: return None

        node.untried_actions.sort(key=lambda x: x[1], reverse=True)
        action, prior = node.untried_actions.pop(0)

        _, _, current_syls, line_idx = self.get_context(node.state)
        target_syls = node.state.get_form()['line_syllables'][line_idx]
        total_lines = len(node.state.get_form()['rhyme_scheme'])
        
        new_text = node.state.text
        if current_syls > 0 and current_syls == target_syls // 2 and random.random() < 0.3:
            new_text = new_text.rstrip() + ", "
            
        if current_syls + get_syllables(action) == target_syls:
            if line_idx == total_lines - 1:
                punctuation = random.choices(['.', '!', '?'], weights=[0.8, 0.15, 0.05])[0]
            else:
                punctuation = random.choices([',', ';', '', ''], weights=[0.4, 0.1, 0.3, 0.2])[0]
            new_text += action + punctuation + "\n"
            
            rhyme_class = node.state.get_form()['rhyme_scheme'][line_idx]
            if rhyme_class and rhyme_class not in node.state.rhyme_map: node.state.rhyme_map[rhyme_class] = action
        else:
            new_text += action + " "

        child_state = PoemState(new_text, node.state.form_name)
        child_state.rhyme_map = node.state.rhyme_map.copy()
        child_c_puct = max(0.5, node.c_puct * 0.8) 
        child_node = Node(child_state, parent=node, action=action, prior=prior, c_puct=child_c_puct)
        node.children.append(child_node)
        return child_node

    def rollout(self, state):
        current_state = PoemState(state.text, state.form_name)
        current_state.rhyme_map = state.rhyme_map.copy()
        
        while not current_state.is_terminal():
            temp_node = Node(current_state)
            temp_node.untried_actions = self.get_valid_actions(temp_node)
            if not temp_node.untried_actions: break 

            _, _, current_syls, line_idx = self.get_context(current_state)
            target_syls = current_state.get_form()['line_syllables'][line_idx]
            total_lines = len(current_state.get_form()['rhyme_scheme'])
            
            safe_actions = [(w, p) for w, p in temp_node.untried_actions if current_syls + get_syllables(w) <= target_syls]
            if not safe_actions: break

            words, priors = zip(*safe_actions)
            action = random.choices(words, weights=priors, k=1)[0]

            if current_syls > 0 and current_syls == target_syls // 2 and random.random() < 0.3:
                current_state.text = current_state.text.rstrip() + ", "

            if current_syls + get_syllables(action) == target_syls:
                if line_idx == total_lines - 1:
                    punctuation = random.choices(['.', '!', '?'], weights=[0.8, 0.15, 0.05])[0]
                else:
                    punctuation = random.choices([',', ';', '', ''], weights=[0.4, 0.1, 0.3, 0.2])[0]
                current_state.text += action + punctuation + "\n"
                rhyme_class = current_state.get_form()['rhyme_scheme'][line_idx]
                if rhyme_class and rhyme_class not in current_state.rhyme_map: current_state.rhyme_map[rhyme_class] = action
            else:
                current_state.text += action + " "

        return current_state.text

    def evaluate(self, poem_text, form_name):
        if (poem_text, form_name) in self.transposition_table: return self.transposition_table[(poem_text, form_name)]

        score = 0.0
        form = POEM_FORMS[form_name]
        weights = form['weights']
        lines = [l.strip() for l in poem_text.split('\n') if l.strip()]
        
        completion_ratio = len(lines) / len(form['rhyme_scheme'])
        score += len(lines) * 25.0 
        
        rhyme_score = 0.0
        meter_score = 0.0
        
        for i, line in enumerate(lines):
            words = line.split()
            if not words: continue

            rhyme_class = form['rhyme_scheme'][i]
            if i > 0 and rhyme_class:
                for j in range(i-1, -1, -1):
                    if form['rhyme_scheme'][j] == rhyme_class:
                        w1 = lines[j].split()[-1].strip('.,!?;:"')
                        w2 = words[-1].strip('.,!?;:"')
                        if do_rhyme(w1, w2):
                            rhyme_score += 150.0
                            if get_concepts(w1).intersection(get_concepts(w2)): rhyme_score += 50.0
                        elif is_slant_rhyme(w1, w2): rhyme_score += 75.0   
                        break

            target_meter = form['line_meters'][i]
            if target_meter:
                line_stress = "".join([get_stress(w) for w in words])
                meter_score += evaluate_meter(line_stress, target_meter)

            score += calculate_alliteration(words)
            score += calculate_lexical_density(words)

        score += (rhyme_score * weights.get("rhyme", 1.0))
        score += (meter_score * weights.get("meter", 1.0))
        score += (calculate_sensory_engagement(lines) * weights.get("sensory", 1.0))
        score += (calculate_narrative_progression(lines) * weights.get("narrative", 1.0))
        score += (calculate_macro_rhythm(lines, form) * weights.get("macro", 1.0))

        score += calculate_thematic_coherence(lines)
        score += calculate_sentiment_arc(lines)
        score += calculate_rhetorical_devices(lines)
        score += calculate_enjambment_variety(lines)

        ttr = calculate_ttr(lines)
        score *= ttr 

        if completion_ratio < 1.0: score *= 0.8 

        self.transposition_table[(poem_text, form_name)] = score
        return score

    def select_best_child(self, node): return max(node.children, key=lambda c: c.puct())

    def run(self):
        for _ in range(self.iterations):
            node = self.root
            while node.children and not node.state.is_terminal():
                if node.untried_actions and len(node.children) < node.max_children: break 
                node = self.select_best_child(node)

            if not node.state.is_terminal():
                expanded = self.expand(node)
                if expanded: node = expanded

            final_poem = node.state.text if node.state.is_terminal() else self.rollout(node.state)
            reward = self.evaluate(final_poem, self.root.state.form_name)

            temp_node = node
            discount = 1.0
            while temp_node is not None:
                temp_node.visits += 1
                temp_node.value += reward * discount
                temp_node = temp_node.parent
                discount *= self.gamma

        node = self.root
        while node.children:
            node = max(node.children, key=lambda c: (c.value / c.visits if c.visits > 0 else 0, c.visits))
            
        raw_poem = node.state.text
        final_score = self.evaluate(raw_poem, self.root.state.form_name)
        
        lines = raw_poem.split('\n')
        formatted_lines = []
        for i, line in enumerate(lines):
            if line.strip():
                line = line.strip()
                line = line[0].upper() + line[1:]
                formatted_lines.append(line)
        
        formatted_poem = '\n'.join(formatted_lines)
        return formatted_poem, final_score

# ==========================================
# Execution: Autonomous Dataset Generation
# ==========================================
if __name__ == "__main__":
    corpus = """
    the night is dark and deep the moon is bright and high 
    the stars shine in the sky the wind begins to sigh 
    the forest stands alone the owl begins to moan 
    the shadows hide the stone i wander far from home
    the moon glows in the dark the dog begins to bark
    i walk across the park and listen to the lark
    the night is cold and still the trees stand on the hill
    the water in the rill flows gently down the hill
    the raven seeks the light the falcon takes to flight
    i stand outside the night my face is pale and white
    the fire burns the wood the hero understood
    the silent brotherhood would help him if they could
    there once was a young man from nantucket who kept all his cash in a bucket
    his daughter named nan ran away with a man and as for the bucket nan took it
    """
    
    nlp_model = NgramModel(corpus)
    db = PoetryDatabase("unsupervised_poetry.db")
    
    # Unsupervised Generation Parameters
    NUM_POEMS_TO_GENERATE = 30
    MIN_ACCEPTABLE_SCORE = 150.0
    forms = list(POEM_FORMS.keys())
    
    print(f"Starting Unsupervised Dataset Generation. Target: {NUM_POEMS_TO_GENERATE} poems.")
    print(f"Minimum acceptable score for dataset inclusion: {MIN_ACCEPTABLE_SCORE}\n")
    
    generated_count = 0
    rejected_count = 0
    
    while generated_count < NUM_POEMS_TO_GENERATE:
        # 1. Autonomously select a form and a seed word
        target_form = random.choice(forms)
        seed_word = nlp_model.get_random_seed()
        
        print(f"[{generated_count+1}/{NUM_POEMS_TO_GENERATE}] Generating {target_form} from seed '{seed_word}'...", end=" ")
        
        # 2. Run MCTS
        mcts = PoemMCTS_Unsupervised(seed_word, nlp_model, form_name=target_form, iterations=1500, gamma=0.9)
        poem, score = mcts.run()
        
        # 3. Quality Filter
        if score >= MIN_ACCEPTABLE_SCORE:
            db.save_poem(seed_word, target_form, poem, score)
            generated_count += 1
            print(f"ACCEPTED (Score: {score:.2f})")
        else:
            rejected_count += 1
            print(f"REJECTED (Score: {score:.2f})")

    print("\n" + "="*50)
    print("DATASET GENERATION COMPLETE")
    print(f"Total Saved to DB: {db.get_count()}")
    print(f"Total Rejected by Quality Filter: {rejected_count}")
    
    # Export to JSONL for ML Training
    db.export_to_jsonl("poetry_dataset.jsonl")