CC = gcc
CFLAGS = -Wall -O2
LDFLAGS = -lasound -lpthread

audio_echo_test: audio_echo_test.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f audio_echo_test

.PHONY: clean
