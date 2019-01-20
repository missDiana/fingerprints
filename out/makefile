SRC=src
INC=include
BIN=bin
SRCS=$(wildcard ${SRC}/*.cpp)
OBJS=$(patsubst ${SRC}/%.cpp, ${BIN}/%.o, ${SRCS})
TARGET_BINARY=${BIN}/util #[replace by target_binary_name]
CXX ?= g++  # or clang++
CXXFLAGS += -I${INC} -std=c++14 -pedantic
CXXFLAGS += -Wall -Wextra -Wno-unused
all: ${TARGET_BINARY}
${TARGET_BINARY} : ${OBJS}
	${CXX} $^ -o $@ `pkg-config --cflags --libs opencv`
${BIN}/%.o: ${SRC}/%.cpp
	${CXX} ${CXXFLAGS} -c $< -o $@ `pkg-config --cflags --libs opencv`
.PHONY: clean
clean:
	rm -f ${OBJS}
	rm -f ${TARGET_BINARY}
