/* __author__ = 'klb3713' */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_STRING 100
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef float real;                    // Precision of float numbers

struct vocab_word {
    long long cn;
    int *point;
    char *word, *code, codelen;
};

char train_file[MAX_STRING], sample_file[MAX_STRING];
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
struct vocab_word *vocab;
int debug_mode = 2, window = 5, min_count = 5, min_reduce = 1;
int *vocab_hash;
long long vocab_max_size = 1000, vocab_size = 0;
long long train_words = 0, file_size = 0;
clock_t start;
int *word_window;


// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin) {
    int a = 0, ch;
    while (!feof(fin)) {
        ch = fgetc(fin);
        if (ch == 13) continue;
        if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
            if (a > 0) {
                if (ch == '\n') ungetc(ch, fin);
                break;
            }
            if (ch == '\n') {
                strcpy(word, (char *)"</s>");
                return;
            } else continue;
        }
        word[a] = ch;
        a++;
        if (a >= MAX_STRING - 1) a--;   // Truncate too long words
    }
    word[a] = 0;
}

// Returns hash value of a word
int GetWordHash(char *word) {
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
    hash = hash % vocab_hash_size;
    return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word) {
    unsigned int hash = GetWordHash(word);
    while (1) {
        if (vocab_hash[hash] == -1) return -1;
        if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
        hash = (hash + 1) % vocab_hash_size;
    }
    return -1;
}


// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin) {
    char word[MAX_STRING];
    ReadWord(word, fin);
    if (feof(fin)) return -1;
    return SearchVocab(word);
}

// Adds a word to the vocabulary
int AddWordToVocab(char *word) {
    unsigned int hash, length = strlen(word) + 1;
    if (length > MAX_STRING) length = MAX_STRING;
    vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
    strcpy(vocab[vocab_size].word, word);
    vocab[vocab_size].cn = 0;
    vocab_size++;
    // Reallocate memory if needed
    if (vocab_size + 2 >= vocab_max_size) {
        vocab_max_size += 1000;
        vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
    }
    hash = GetWordHash(word);
    while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
    vocab_hash[hash] = vocab_size - 1;
    return vocab_size - 1;
}

// Used later for sorting by word counts
int VocabCompare(const void *a, const void *b) {
    return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
void SortVocab() {
    int a, size;
    unsigned int hash;
    // Sort the vocabulary and keep </s> at the first position
    qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    size = vocab_size;
    train_words = 0;
    for (a = 0; a < size; a++) {
        // Words occuring less than min_count times will be discarded from the vocab
        if (vocab[a].cn < min_count) {
            vocab_size--;
            free(vocab[vocab_size].word);
        } else {
            // Hash will be re-computed, as after the sorting it is not actual
            hash=GetWordHash(vocab[a].word);
            while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
            vocab_hash[hash] = a;
            train_words += vocab[a].cn;
        }
    }
    vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
    // Allocate memory for the binary tree construction
    for (a = 0; a < vocab_size; a++) {
        vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
        vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
    }
}

// Reduces the vocabulary by removing infrequent tokens
void ReduceVocab() {
    int a, b = 0;
    unsigned int hash;
    for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
        vocab[b].cn = vocab[a].cn;
        vocab[b].word = vocab[a].word;
        b++;
    } else free(vocab[a].word);
    vocab_size = b;
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    for (a = 0; a < vocab_size; a++) {
        // Hash will be re-computed, as it is not actual
        hash = GetWordHash(vocab[a].word);
        while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
        vocab_hash[hash] = a;
    }
    fflush(stdout);
    min_reduce++;
}

void LearnVocabFromTrainFile() {
    char word[MAX_STRING];
    FILE *fin;
    long long a, i;
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    fin = fopen(train_file, "rb");
    if (fin == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    vocab_size = 0;
    AddWordToVocab((char *)"</s>");
    while (1) {
        ReadWord(word, fin);
        if (feof(fin)) break;
        train_words++;
        if ((debug_mode > 1) && (train_words % 100000 == 0)) {
            printf("%lldK%c", train_words / 1000, 13);
            fflush(stdout);
        }
        i = SearchVocab(word);
        if (i == -1) {
            a = AddWordToVocab(word);
            vocab[a].cn = 1;
        } else vocab[i].cn++;
        if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
    }
    SortVocab();
    if (debug_mode > 0) {
        printf("Vocab size: %lld\n", vocab_size);
        printf("Words in train file: %lld\n", train_words);
    }
    file_size = ftell(fin);
    fclose(fin);
}

void SaveVocab() {
    long long i;
    FILE *fo = fopen(save_vocab_file, "wb");
    for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
    fclose(fo);
}

void ReadVocab() {
    long long a, i = 0;
    char c;
    char word[MAX_STRING];
    FILE *fin = fopen(read_vocab_file, "rb");
    if (fin == NULL) {
        printf("Vocabulary file not found\n");
        exit(1);
    }
    for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
    vocab_size = 0;
    while (1) {
        ReadWord(word, fin);
        if (feof(fin)) break;
        a = AddWordToVocab(word);
        fscanf(fin, "%lld%c", &vocab[a].cn, &c);
        i++;
    }
    SortVocab();
    if (debug_mode > 0) {
        printf("Vocab size: %lld\n", vocab_size);
        printf("Words in train file: %lld\n", train_words);
    }
    fin = fopen(train_file, "rb");
    if (fin == NULL) {
        printf("ERROR: training data file not found!\n");
        exit(1);
    }
    fseek(fin, 0, SEEK_END);
    file_size = ftell(fin);
    fclose(fin);
}

void build_vocabulary() {
    vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
    vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));

    if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
    if (save_vocab_file[0] != 0) SaveVocab();
}

void build_dataset() {
    printf("Build samples to %s ...\n", sample_file);
    int i, index;
    long long sample_size = 0;
    FILE *fo = fopen(sample_file, "wb");
    FILE *fi = fopen(train_file, "rb");
    word_window = (int *)calloc(window, sizeof(int));
    int window_index = 0;
    int half_window = window / 2;
    int unknown = 0;
    int ignore = 0;
    int first_index = 0;

    while (window_index < window) {
        index = ReadWordIndex(fi);
        if (feof(fi)) break;
        if (index == -1)    word_window[window_index] = 0;
        else word_window[window_index] = index;
        window_index++;
    }

    window_index == 0;
    while (1) {
        if (feof(fi)) break;

        unknown = 0;
        ignore = 0;

        for (i = 0; i < window; i++) {
            if (word_window[i] == 0)    unknown++;
            if (unknown > half_window) {
                ignore = 1;
                break;
            }
        }
        if (!ignore) {
            first_index = window_index;
            for (i = 0; i < window; i++) {
                fprintf(fo, "%d ", word_window[first_index]);
                first_index++;
                first_index = first_index % window;
            }
            fprintf(fo, "\n");
            sample_size++;
        }

        index = ReadWordIndex(fi);
        if (index == -1)    word_window[window_index] = 0;
        else word_window[window_index] = index;
        window_index++;
        window_index = window_index % window;
    }

    printf("Samples size: %lld\n", sample_size);
    fclose(fi);
    fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
    int a;
    for (a = 1; a < argc; a++) {
        if (!strcmp(str, argv[a])) {
            if (a == argc - 1) {
                printf("Argument missing for %s\n", str);
                exit(1);
            }
            return a;
        }
    }
    return -1;
}

int main(int argc, char **argv) {
    int i;
    if (argc == 1) {
        printf("Build Samples file for model.\n\n");
        printf("Options:\n");
        printf("\t-train <file>\n");
        printf("\t\tUse text data from <file> to train the model\n");
        printf("\t-window <int>\n");
        printf("\t\tSet max skip length between words; default is 5\n");
        printf("\t-sample <file>\n");
        printf("\t\tThe samples will be saved to <file>\n");
        printf("\t-min-count <int>\n");
        printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
        printf("\t-debug <int>\n");
        printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
        printf("\t-save-vocab <file>\n");
        printf("\t\tThe vocabulary will be saved to <file>\n");
        printf("\t-read-vocab <file>\n");
        printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
        printf("\nExamples:\n");
        printf("./build_dataset -train data.txt -sample samples -debug 2 -window 5 -save-vocab vocabulary.txt \n\n");
        return 0;
    }

    sample_file[0] = 0;
    save_vocab_file[0] = 0;
    read_vocab_file[0] = 0;
    if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) strcpy(sample_file, argv[i + 1]);
    if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
    if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);

    build_vocabulary();
    build_dataset();

    return 0;

}
