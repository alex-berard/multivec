CC = g++
DEBUG = -g
CCFLAGS = -c $(DEBUG) -O3 -std=c++0x -Wall -Wno-sign-compare
LFLAGS = $(DEBUG) -pthread -lm -lboost_program_options -lboost_serialization

SOURCE = %.cpp
HEADER = %.h

BUILD = build

all: bivec word2vec

$(BUILD)/%.o: $(SOURCE)
	mkdir -p $(BUILD) && $(CC) -c -o $@ $< $(CCFLAGS)

bivec: $(BUILD)/bivec_main.o $(BUILD)/bivec.o $(BUILD)/word2vec.o
	$(CC) -o $@ $^ $(LFLAGS)

word2vec: $(BUILD)/word2vec_main.o $(BUILD)/word2vec.o
	$(CC) -o $@ $^ $(LFLAGS)

clean:
	rm -rf build bivec word2vec
