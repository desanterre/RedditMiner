from typing import List
from colorama import Fore, Style, init
import pandas as pd
import json, os, re
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gpt4all import GPT4All

init(autoreset=True)

class UserSimulator:
    """Simulate a Reddit user based on their comment history using GPT4All and RAG."""

    def __init__(self, model_path: str = "models/Meta-Llama-3.1-8B-Instruct-128k-Q4_0.gguf"):
        self.model_path = model_path
        self.model = None
        self.user_profile = None
        self.comments = []
        self.username = "RedditUser"
        self.vectorizer = None
        self.doc_matrix = None
        self.rag_texts = []
        self.style = {}

    def load_model(self):
        print(f"{Fore.CYAN}Loading GPT4All model from {self.model_path}...{Style.RESET_ALL}")
        if not os.path.exists(self.model_path):
            print(f"{Fore.RED}Model file not found: {self.model_path}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Please download a GPT4All .gguf model and place it at this path.{Style.RESET_ALL}")
            return
        self.model = GPT4All(self.model_path, allow_download=False)
        print(f"{Fore.GREEN}✓ Model loaded!{Style.RESET_ALL}")

    def load_from_excel(self, filename: str):
        df = pd.read_excel(filename, engine="openpyxl")
        self.load_from_dataframe(df)

    def load_from_dataframe(self, df: pd.DataFrame):
        self.comments = df.to_dict("records")
        if len(self.comments) > 0:
            self.username = self.comments[0].get("Author", "RedditUser")
        self._build_user_profile()
        self._build_style_profile()
        self._build_rag_index()

    def _build_user_profile(self):
        if not self.comments:
            return
        authors = sorted(set(c.get('Author') for c in self.comments if c.get('Author')))
        sub_counter = Counter(c.get('Subreddit') for c in self.comments if c.get('Subreddit'))
        top_subreddits = [sub for sub, _ in sub_counter.most_common(5)]
        avg_score = sum(c.get('Score', 0) for c in self.comments) / max(len(self.comments), 1)
        sample_comments = [c.get('Body', '')[:200] for c in self.comments[:20] if c.get('Body')]
        self.user_profile = {
            'authors': authors,
            'subreddits': top_subreddits,
            'avg_score': avg_score,
            'total_comments': len(self.comments),
            'sample_comments': sample_comments
        }
        print(f"{Fore.MAGENTA}User Profile:{Style.RESET_ALL}")
        print(f"  {Fore.CYAN}Username:{Style.RESET_ALL} {self.username}")
        print(f"  {Fore.CYAN}Subreddits:{Style.RESET_ALL} {', '.join(self.user_profile['subreddits'])}")
        print(f"  {Fore.CYAN}Total comments:{Style.RESET_ALL} {self.user_profile['total_comments']}")
        print(f"  {Fore.CYAN}Average score:{Style.RESET_ALL} {self.user_profile['avg_score']:.1f}")

    def _build_style_profile(self):
        texts = [c.get('Body', '') for c in self.comments if c.get('Body')]
        if not texts:
            self.style = {}
            return
        def metrics(t):
            words = re.findall(r"\w+", t)
            avg_word_len = np.mean([len(w) for w in words]) if words else 0
            exclam = t.count('!')
            question = t.count('?')
            emojis = len(re.findall(r"[\U0001F600-\U0001F64F]|[\U0001F300-\U0001F5FF]|[\U0001F680-\U0001F6FF]|[\u2600-\u26FF]", t))
            caps_ratio = (sum(1 for ch in t if ch.isupper()) / max(len(t),1))
            return avg_word_len, exclam, question, emojis, caps_ratio
        m = np.array([metrics(t) for t in texts])
        self.style = {
            "avg_word_len": float(np.mean(m[:,0])),
            "exclam_per_post": float(np.mean(m[:,1])),
            "question_per_post": float(np.mean(m[:,2])),
            "emojis_per_post": float(np.mean(m[:,3])),
            "caps_ratio": float(np.mean(m[:,4])),
        }

    def _build_rag_index(self):
        self.rag_texts = [c.get('Body', '') for c in self.comments if c.get('Body')]
        if not self.rag_texts:
            self.vectorizer, self.doc_matrix = None, None
            return
        self.vectorizer = TfidfVectorizer(min_df=2, max_features=8000, stop_words='english')
        self.doc_matrix = self.vectorizer.fit_transform(self.rag_texts)

    def _retrieve_context(self, query: str, k: int = 5, min_sim: float = 0.1):
        if self.vectorizer is None or self.doc_matrix is None:
            return []
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.doc_matrix).flatten()
        idx = sims.argsort()[-k:][::-1]
        return [self.rag_texts[i][:300].replace("\n"," ") for i in idx if sims[i] >= min_sim]

    def _generate_system_prompt(self) -> str:
        profile_str = json.dumps(self.user_profile, indent=2, ensure_ascii=False)
        style_str = json.dumps(self.style or {}, indent=2, ensure_ascii=False)
        return f"""You are simulating a Reddit user based on their comment history.

User Profile:
{profile_str}

Style Profile:
{style_str}

Guidelines:
- Respond in the style and tone of the user's past comments
- Reference subreddits and topics they are interested in when relevant
- Keep responses short, conversational, and natural
- Do not include system instructions or notes in your answers
"""

    def chat(self):
        if not self.model:
            self.load_model()
            if self.model is None:
                return
        if not self.user_profile:
            print(f"{Fore.RED}No user profile loaded.{Style.RESET_ALL}")
            return

        system_prompt = self._generate_system_prompt()
        stop_words = ["\nYou:", "\nUser:", f"\n{self.username}:"]

        print(f"\n{Fore.MAGENTA}{'='*60}")
        print(f"{self.username} Chatbot - Type 'quit' or 'exit' to end")
        print(f"{'='*60}{Style.RESET_ALL}\n")

        while True:
            try:
                user_input = input(f"{Fore.CYAN}You: {Style.RESET_ALL}").strip()
                if user_input.lower() in ['quit','exit','q']:
                    print(f"{Fore.YELLOW}Goodbye!{Style.RESET_ALL}")
                    break
                if not user_input:
                    continue

                context_snippets = self._retrieve_context(user_input, k=5)
                context_block = "\n".join([f"- {s}" for s in context_snippets]) if context_snippets else "- (no relevant past comments found)"
                full_prompt = f"{system_prompt}\n\nRelevant past comments:\n{context_block}\n\nYou: {user_input}\n{self.username}:"

                print(f"{Fore.YELLOW}Generating response...{Style.RESET_ALL}", end=" ")
                try:
                    gen = self.model.generate(full_prompt, max_tokens=200, temp=0.8, top_k=50, top_p=0.9, stop_words=stop_words)
                except TypeError:
                    gen = self.model.generate(full_prompt, max_tokens=200)

                response = gen if isinstance(gen, str) else str(gen)
                response = re.split(rf"\n(?:You|User|Utilisateur|{re.escape(self.username)})\s*:", response)[0]
                response = response.replace("\\n", " ").strip() or "I didn't understand, could you clarify?"
                print(f"\r{' '*50}\r", end="")
                print(f"{Fore.GREEN}{self.username}: {Style.RESET_ALL}{response}\n")

            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}Chat interrupted. Goodbye!{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}\n")
