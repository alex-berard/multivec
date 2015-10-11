CC = g++
DEBUG = -g
CCFLAGS = -c $(DEBUG) -O3 -std=c++0x -Wall -Wno-sign-compare
LFLAGS = $(DEBUG) -pthread -lm -lboost_program_options -lboost_serialization

SOURCE = %.cpp
HEADER = %.h

BUILD = build

all: word2vec compute-accuracy

$(BUILD)/%.o: $(SOURCE)
	mkdir -p $(BUILD) && $(CC) -c -o $@ $< $(CCFLAGS)

word2vec: $(BUILD)/word2vec_main.o $(BUILD)/word2vec.o
	$(CC) -o $@ $^ $(LFLAGS)

compute-accuracy: $(BUILD)/compute-accuracy.o
	$(CC) -o $@ $^ $(LFLAGS)
#compute-accuracy:
	#$(CC) compute-accuracy.c -o compute-accuracy -lm

clean:
	rm -rf build word2vec compute-accuracy
