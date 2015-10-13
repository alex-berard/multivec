#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include <ctype.h>
#include <time.h>

const long long max_w = 50;              // max length of vocabulary entries

int main(int argc, char **argv)
{
    if (argc < 1) {
        printf("Usage: ./compute-accuracy <FILE>\nwhere FILE contains word projections\n");
        return 0;
    }

    FILE* f = fopen(argv[1], "rb");
    if (f == NULL) {
        printf("Input file not found\n");
        return -1;
    }

    long long words, size;

    fscanf(f, "%lld", &words);
    fscanf(f, "%lld", &size);

    char *vocab = (char *)malloc(words * max_w * sizeof(char));
    float *M = (float *)malloc(words * size * sizeof(float));
    if (M == NULL) {
        printf("Cannot allocate memory: %lld MB\n", words * size * sizeof(float) / 1048576);
        return -1;
    }

    for (int b = 0; b < words; b++) {
        int a = 0;
        while (1) {
            vocab[b * max_w + a] = fgetc(f);
            if (feof(f) || (vocab[b * max_w + a] == ' ')) break;
            if ((a < max_w) && (vocab[b * max_w + a] != '\n')) a++;
        }
        vocab[b * max_w + a] = 0;
        for (a = 0; a < max_w; a++) vocab[b * max_w + a] = toupper(vocab[b * max_w + a]);
        for (a = 0; a < size; a++) fread(&M[a + b * size], sizeof(float), 1, f);
        int len = 0;
        for (a = 0; a < size; a++) len += M[a + b * size] * M[a + b * size];
        len = sqrt(len);
        for (a = 0; a < size; a++) M[a + b * size] /= len;
    }
    fclose(f);

    int n = 100;

    float min_distance = 1;
    float max_distance = -1;

    srand(time(NULL));

    for (int i = 0; i < n; ++i) {
        int word1 = rand() % words; // for n random words

        for (int word2 = 0; word2 < words; ++word2) { // compute their cosine similarity with all other words
            if (word2 == word1) continue;

            float distance = 0;
            float norm_x = 0;
            float norm_y = 0;

            for (int k = 0; k < size; ++k) {
                float x = M[k + word2 * size];
                float y = M[k + word1 * size];

                norm_x += x * x;
                norm_y += y * y;
                distance += x * y;
            } // sim(v, w) = dot(v, w) / (norm(v) * norm(w))

            distance /= sqrt(norm_x) * sqrt(norm_y);

            if (distance < min_distance) {
                min_distance = distance;
            } else if (distance > max_distance) {
                max_distance = distance;
            }
        }
    }

    printf("Min distance: %f\n", min_distance);
    printf("Max distance: %f\n", max_distance);

    return 0;
}
