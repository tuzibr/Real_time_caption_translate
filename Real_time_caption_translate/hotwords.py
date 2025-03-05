import Levenshtein

def correct_sentence(sentence, hotwords, threshold_factor=0.3):
    words = sentence.split()
    corrected_words = []
    i = 0
    while i < len(words):
        # Check two-word phrases if possible
        if i < len(words) - 1:
            phrase = words[i] + " " + words[i + 1]
            # Only compare with hotwords that are phrases (contain a space)
            phrase_hotwords = [hw for hw in hotwords if " " in hw]
            if phrase_hotwords:
                distances = [(hw, Levenshtein.distance(phrase, hw)) for hw in phrase_hotwords]
                best_match, min_distance = min(distances, key=lambda x: x[1])
                if min_distance <= threshold_factor * len(phrase):
                    corrected_words.append(best_match)
                    i += 2  # Skip the next word since we used two
                    continue
            # If no phrase match, check if combined phrase matches single-word hotword
            combined = words[i] + words[i + 1]
            single_hotwords = [hw for hw in hotwords if " " not in hw]
            if single_hotwords:
                distances = [(hw, Levenshtein.distance(combined, hw)) for hw in single_hotwords]
                best_match, min_distance = min(distances, key=lambda x: x[1])
                if min_distance <= threshold_factor * len(combined):
                    corrected_words.append(best_match)
                    i += 2  # Skip the next word since we used two
                    continue
        # If no phrase or combined match, process single word
        word = words[i]
        if word in hotwords:
            corrected_words.append(word)
        else:
            distances = [(hw, Levenshtein.distance(word, hw)) for hw in hotwords]
            best_match, min_distance = min(distances, key=lambda x: x[1])
            if min_distance <= threshold_factor * len(word):
                corrected_words.append(best_match)
            else:
                corrected_words.append(word)
        i += 1
    return " ".join(corrected_words)

if __name__ == "__main__":

    sentence = "in september of two thousand and nineteen dallas police officer a m b e r geiger was sentenced for murder"
    hotwords = ["amber", "world", "test"]
    corrected_sentence = correct_sentence(sentence, hotwords)
    print(corrected_sentence)