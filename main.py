import random
import string
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import time
from collections import Counter
import json
import os
import statistics
import torch
import numpy as np


# Check if MPS (Metal Performance Shaders) is available
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()

# Pre-compile the word table into a set for faster lookups
WORD_TABLE = {'void', 'infinite', 'help', 'me', 'the', 'and', 'baker', 'exit', 'leave'}

# Convert symbols to a tuple for better performance
SYMBOLS = (
    '▓', '░', '■', '□', '⬿', '⬧', '⬨', '✧', '🜅', '⬷', '⬽', '🜋', '⬱', '⬦', '⬯', '⬵', '⬪', '⬤', '⬢', '⬡', '⬥', '★', '☆',
    '✦')

elapsed_time = 0
attempts_per_second = 0

# Pre-generate valid characters
VALID_CHARS = tuple(string.ascii_lowercase)

# Add common English letter frequencies for scoring
LETTER_FREQUENCIES = {
    'e': 12.7, 't': 9.1, 'a': 8.2, 'o': 7.5, 'i': 7.0, 'n': 6.7, 's': 6.3,
    'h': 6.1, 'r': 6.0, 'd': 4.3, 'l': 4.0, 'c': 2.8, 'u': 2.8, 'm': 2.4,
    'w': 2.4, 'f': 2.2, 'g': 2.0, 'y': 2.0, 'p': 1.9, 'b': 1.5, 'v': 1.0,
    'k': 0.8, 'j': 0.15, 'x': 0.15, 'q': 0.10, 'z': 0.07
}

# Convert letter frequencies to tensor
LETTER_FREQ_TENSOR = torch.zeros(26, device=DEVICE)
for char, freq in LETTER_FREQUENCIES.items():
    LETTER_FREQ_TENSOR[ord(char) - ord('a')] = freq


def save_mappings(valid_mappings, filename='decoder_checkpoint.json'):
    """Save valid mappings to a JSON file"""
    serializable_mappings = []
    for mapping, decoded, score, words in valid_mappings:
        serializable_mappings.append({
            'mapping': {k: v for k, v in mapping.items()},
            'decoded': decoded,
            'score': score,
            'words': words
        })

    with open(filename, 'w') as f:
        json.dump(serializable_mappings, f)


def load_mappings(filename='decoder_checkpoint.json'):
    """Load valid mappings from a JSON file"""
    if not os.path.exists(filename):
        return []

    with open(filename, 'r') as f:
        data = json.load(f)

    valid_mappings = []
    for item in data:
        mapping = {k: v for k, v in item['mapping'].items()}
        valid_mappings.append((mapping, item['decoded'], item['score'], item['words']))

    return valid_mappings


def calculate_score(decoded_text):
    """
    Calculate score for CPU version
    """
    # Word importance weights
    WORD_WEIGHTS = {
        'void': 5.0,
        'baker': 5.0,
        'infinite': 5.0,
        'help': 2.0,
        'exit': 2.0,
        'leave': 2.0,
        'the': 0.5,
        'and': 0.5,
        'me': 0.3
    }

    MIN_FULL_SCORE_LENGTH = 4
    score = 0
    found_words = []
    word_counts = Counter()

    text_lower = decoded_text.lower()
    for word in WORD_TABLE:
        count = text_lower.count(word)
        if count > 0:
            if len(word) < MIN_FULL_SCORE_LENGTH:
                effective_count = np.log2(count + 1)
            else:
                effective_count = count

            word_score = (len(word) * effective_count * 20) * WORD_WEIGHTS.get(word, 1.0)

            if len(word) <= 2 and count > 3:
                word_score *= 0.5

            score += word_score
            found_words.extend([word] * count)
            word_counts[word] = count

    # Check for consecutive words
    if found_words:
        words_str = ' '.join(found_words)
        for word1 in WORD_TABLE:
            for word2 in WORD_TABLE:
                if len(word1) >= MIN_FULL_SCORE_LENGTH and len(word2) >= MIN_FULL_SCORE_LENGTH:
                    if f"{word1} {word2}" in words_str:
                        score *= 1.5

    # Bonus for multiple unique words
    unique_words = set(found_words)
    if len(unique_words) > 1:
        length_weighted_bonus = sum(len(word) * WORD_WEIGHTS.get(word, 1.0)
                                    for word in unique_words) / len(unique_words)
        score *= (1 + (len(unique_words) - 1) * length_weighted_bonus * 0.5)

    # Penalty for short words
    short_word_ratio = sum(1 for w in found_words if len(w) <= 2) / (len(found_words) + 1)
    if short_word_ratio > 0.5:
        score *= (1 - (short_word_ratio - 0.5))

    return score, found_words


class GPUDecoder:
    def __init__(self, cipher_text, batch_size=65536):
        self.cipher_text = cipher_text
        self.batch_size = batch_size
        self.unique_symbols = tuple(set(symbol for symbol in cipher_text if symbol in SYMBOLS))
        self.symbol_to_idx = {symbol: idx for idx, symbol in enumerate(self.unique_symbols)}

        self.cipher_indices = torch.tensor([
            self.symbol_to_idx[c] if c in self.symbol_to_idx else -1
            for c in cipher_text
        ], device=DEVICE)

        self.symbol_positions = {
            symbol: torch.tensor([i for i, c in enumerate(cipher_text) if c == symbol],
                                 device=DEVICE)
            for symbol in self.unique_symbols
        }

        self.char_indices = torch.tensor([ord(c) - ord('a') for c in VALID_CHARS], device=DEVICE)

    def generate_batch_mappings(self):
        n_symbols = len(self.unique_symbols)
        mappings = torch.zeros((self.batch_size, n_symbols), dtype=torch.long, device=DEVICE)
        for i in range(self.batch_size):
            perm = torch.randperm(len(VALID_CHARS), device=DEVICE)[:n_symbols]
            mappings[i] = self.char_indices[perm]
        return mappings

    def decode_batch(self, mappings):
        batch_size = mappings.shape[0]
        text_length = len(self.cipher_text)
        decoded = torch.full((batch_size, text_length), -1,
                             dtype=torch.long, device=DEVICE)
        for symbol_idx, positions in enumerate(self.symbol_positions.values()):
            decoded[:, positions] = mappings[:, symbol_idx]
        return decoded

    def calculate_letter_frequencies(self, decoded_texts):
        batch_size = decoded_texts.shape[0]
        freqs = torch.zeros((batch_size, 26), device=DEVICE)
        for i in range(26):
            freqs[:, i] = (decoded_texts == i).float().sum(dim=1)
        totals = freqs.sum(dim=1, keepdim=True)
        freqs = (freqs / totals) * 100
        freq_scores = -torch.abs(freqs - LETTER_FREQ_TENSOR).sum(dim=1)
        return freq_scores

    def find_words(self, decoded_texts):
        decoded_cpu = decoded_texts.cpu().numpy()
        batch_results = []

        WORD_WEIGHTS = {
            'void': 5.0,
            'baker': 5.0,
            'infinite': 5.0,
            'help': 2.0,
            'exit': 2.0,
            'leave': 2.0,
            'the': 0.5,
            'and': 0.5,
            'me': 0.3
        }

        MIN_FULL_SCORE_LENGTH = 4

        for decoded in decoded_cpu:
            text = ''.join(chr(i + ord('a')) if i != -1 else ' '
                           for i in decoded)

            score = 0
            found_words = []
            word_counts = Counter()

            for word in WORD_TABLE:
                count = text.count(word)
                if count > 0:
                    if len(word) < MIN_FULL_SCORE_LENGTH:
                        effective_count = np.log2(count + 1)
                    else:
                        effective_count = count

                    word_score = (len(word) * effective_count * 20) * WORD_WEIGHTS.get(word, 1.0)

                    if len(word) <= 2 and count > 3:
                        word_score *= 0.5

                    score += word_score
                    found_words.extend([word] * count)
                    word_counts[word] = count

            if found_words:
                words_str = ' '.join(found_words)
                for word1 in WORD_TABLE:
                    for word2 in WORD_TABLE:
                        if len(word1) >= MIN_FULL_SCORE_LENGTH and len(word2) >= MIN_FULL_SCORE_LENGTH:
                            if f"{word1} {word2}" in words_str:
                                score *= 1.5

            unique_words = set(found_words)
            if len(unique_words) > 1:
                length_weighted_bonus = sum(len(word) * WORD_WEIGHTS.get(word, 1.0)
                                            for word in unique_words) / len(unique_words)
                score *= (1 + (len(unique_words) - 1) * length_weighted_bonus * 0.5)

            short_word_ratio = sum(1 for w in found_words if len(w) <= 2) / (len(found_words) + 1)
            if short_word_ratio > 0.5:
                score *= (1 - (short_word_ratio - 0.5))

            batch_results.append((score, found_words))

        return batch_results


def try_decode_batch(args):
    """CPU batch processing"""
    cipher_text, batch_size = args
    valid_mappings = []
    unique_symbols = tuple(set(symbol for symbol in cipher_text if symbol in SYMBOLS))

    symbol_positions = {symbol: [i for i, c in enumerate(cipher_text) if c == symbol]
                        for symbol in unique_symbols}

    for _ in range(batch_size):
        char_sample = random.sample(VALID_CHARS, len(unique_symbols))
        mapping = dict(zip(unique_symbols, char_sample))

        decoded = ''.join(mapping.get(c, c) for c in cipher_text)
        score, found_words = calculate_score(decoded)

        if score > 0:
            valid_mappings.append((mapping, decoded, score, found_words))

    return valid_mappings


def try_decode_batch_gpu(decoder):
    """GPU-accelerated batch processing"""
    mappings = decoder.generate_batch_mappings()
    decoded_texts = decoder.decode_batch(mappings)
    freq_scores = decoder.calculate_letter_frequencies(decoded_texts)
    batch_results = decoder.find_words(decoded_texts)

    valid_mappings = []
    for i, (word_score, found_words) in enumerate(batch_results):
        if word_score > 0:
            mapping = {
                symbol: chr(mappings[i, idx].item() + ord('a'))
                for symbol, idx in decoder.symbol_to_idx.items()
            }

            decoded = ''.join(mapping.get(c, c) for c in decoder.cipher_text)
            final_score = word_score + freq_scores[i].item()
            valid_mappings.append((mapping, decoded, final_score, found_words))

    return valid_mappings


def generate_mappings_parallel(cipher_text, target_count=20000, batch_size=65536, min_score=1000, checkpoint_interval=5):
    """CPU parallel processing version"""
    start_time = time.time()
    last_checkpoint_time = start_time

    valid_mappings = load_mappings()
    print(f"Loaded {len(valid_mappings)} existing mappings")

    total_attempts = 0
    num_processes = mp.cpu_count()

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        while len(valid_mappings) < target_count:
            batches = [(cipher_text, batch_size) for _ in range(num_processes)]
            results = executor.map(try_decode_batch, batches)

            for batch_results in results:
                valid_mappings.extend([r for r in batch_results if r[2] >= min_score])
                total_attempts += batch_size * num_processes

            valid_mappings.sort(key=lambda x: x[2], reverse=True)
            valid_mappings = valid_mappings[:target_count]

            current_time = time.time()
            if current_time - last_checkpoint_time >= checkpoint_interval:
                save_mappings(valid_mappings)
                last_checkpoint_time = current_time
            global elapsed_time
            elapsed_time = current_time - start_time
            global attempts_per_second
            attempts_per_second = total_attempts / elapsed_time
            print(f"\rFound {len(valid_mappings)}/{target_count} valid mappings. "
                  f"Speed: {attempts_per_second:.2f} attempts/second. "
                  f"Best score: {valid_mappings[0][2] if valid_mappings else 0:.1f}", end='')

            if len(valid_mappings) >= target_count and valid_mappings[0][2] >= min_score:
                break

    save_mappings(valid_mappings)

    print(f"\n\nCompleted in {elapsed_time:.2f} seconds")
    print(f"Total attempts: {total_attempts}")
    print(f"Average speed: {attempts_per_second:.2f} attempts/second")

    return valid_mappings


def analyze_results(valid_mappings):
    """Analyze and print detailed statistics about the results"""
    if not valid_mappings:
        print("No valid mappings found to analyze.")
        return

    print("\nDetailed Result Analysis:")
    print("=" * 50)

    # Word frequency analysis
    all_found_words = []
    word_lengths = []
    for _, _, _, words in valid_mappings:
        all_found_words.extend(words)
        word_lengths.extend(len(word) for word in words)

    word_freq = Counter(all_found_words)
    print("\n1. Word Frequency Analysis:")
    print("-" * 30)
    for word, count in word_freq.most_common():
        percentage = (count / len(all_found_words)) * 100
        print(f"- '{word}' ({len(word)} letters): found {count} times ({percentage:.1f}% of all words)")

    # Word length analysis
    avg_word_length = sum(word_lengths) / len(word_lengths) if word_lengths else 0
    print(f"\nAverage word length: {avg_word_length:.1f} letters")
    print(f"Word length distribution:")
    length_freq = Counter(word_lengths)
    for length in sorted(length_freq.keys()):
        print(f"- {length} letters: {length_freq[length]} words")

    # Score statistics
    scores = [score for _, _, score, _ in valid_mappings]
    print("\n2. Score Statistics:")
    print("-" * 30)
    print(f"- Highest score: {max(scores):.1f}")
    print(f"- Average score: {statistics.mean(scores):.1f}")
    print(f"- Median score: {statistics.median(scores):.1f}")
    print(f"- Lowest score: {min(scores):.1f}")
    print(f"- Standard deviation: {statistics.stdev(scores):.1f}")

    # Symbol mapping analysis
    print("\n3. Symbol Mapping Analysis:")
    print("-" * 30)
    symbol_mappings = {}
    for mapping, _, _, _ in valid_mappings:
        for symbol, char in mapping.items():
            if symbol not in symbol_mappings:
                symbol_mappings[symbol] = Counter()
            symbol_mappings[symbol][char] += 1

    for symbol, char_counts in symbol_mappings.items():
        most_common = char_counts.most_common(3)
        total = sum(char_counts.values())
        print(f"\nSymbol '{symbol}' most commonly maps to:")
        for char, count in most_common:
            percentage = (count / total) * 100
            print(f"- '{char}': {count} times ({percentage:.1f}%)")

    # Short word ratio analysis
    print("\n4. Short Word Analysis:")
    print("-" * 30)
    for _, decoded, score, words in valid_mappings[:5]:
        short_words = sum(1 for w in words if len(w) <= 2)
        short_ratio = short_words / len(words) if words else 0
        print(f"\nDecoded text: {decoded[:50]}...")
        print(f"Short word ratio: {short_ratio:.1f}")
        print(f"Score: {score:.1f}")


if __name__ == '__main__':
    # Example cipher text
    cipher_text = "▓ ░ ■ □ ⬿⬧🜅⬷⬽🜋⬱⬦⬯⬵⬽⬿⬯⬦⬧⬪⬽⬧⬧⬧⬽⬿⬪⬧⬯⬧⬿⬪⬽⬵⬯⬵⬯⬷⬯⬵⬯⬽⬯⬦⬯⬷⬽⬧⬯⬿⬵⬽⬧⬯⬧⬵⬿⬵⬯⬷⬧⬯⬽⬧⬽⬯⬧⬷⬵⬯⬵⬧⬷⬿⬯⬦⬵⬧⬯⬵⬽⬯⬵⬽⬧⬿⬯⬷⬵⬧⬷⬯⬧⬯⬽⬿⬷⬽⬯⬷⬦⬯⬧⬷⬽⬧⬿⬧⬽⬿⬵⬯⬷⬯⬵⬧⬷⬽⬯⬷⬦⬯⬧⬦⬿⬷⬿⬽⬯⬦⬯"

    print(f"Using device: {DEVICE}")
    print("Starting GPU-accelerated decoder with enhanced scoring...")

    # Use GPU version if available, otherwise fall back to CPU
    if DEVICE.type in ['cuda', 'mps']:
        valid_mappings = generate_mappings_parallel(cipher_text)
    else:
        print("GPU not available, falling back to CPU version...")
        valid_mappings = generate_mappings_parallel(cipher_text)

    print("\nTop 5 highest-scoring mappings:")
    for i, (mapping, decoded, score, words) in enumerate(valid_mappings[:5], 1):
        print(f"\n{i}. Score: {score:.1f}")
        print(f"   Words found: {', '.join(words)}")
        print(f"   Decoded message: {decoded}")
        print(f"   Mapping: {mapping}")

    # Analyze results
    analyze_results(valid_mappings)