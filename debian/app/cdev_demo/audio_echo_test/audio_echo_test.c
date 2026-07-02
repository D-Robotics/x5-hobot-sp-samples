/*
 * Audio Driver HAT REV2 loopback demo (two phases).
 * Phase 1: record 8ch voice (16 kHz, 16-bit, 5 s).
 * Phase 2: play that clip twice (2 s gap) while capturing the full 8ch stream.
 */
#include <alsa/asoundlib.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHANNELS       8
#define RATE           16000
#define FORMAT         SND_PCM_FORMAT_S16_LE
#define PERIOD_FRAMES  512
#define BUFFER_PERIODS 4
#define CAPTURE_DEV    "plughw:0,1"
#define PLAYBACK_DEV   "plughw:0,0"
#define DURATION_SEC   5
#define GAP_SEC        2
#define PLAY_REPEAT    2
#define PASS_THRESHOLD 500
#define OUT_RECORD     "record_first.wav"
#define OUT_LOOPBACK   "audio_echo_test.wav"
#define HAT_CARD_NAME  "duplex-audio-i2s1"
#define HAT_PRODUCT    "Audio Driver HAT REV2"
#define ASOUND_CARDS   "/proc/asound/cards"
#define HAT_CARD_INDEX 0

static volatile int g_stop;
static volatile int g_err;

static int hat_find_card_index(void)
{
	FILE *f = fopen(ASOUND_CARDS, "r");
	char line[256];
	int card = -1;

	if (!f)
		return -1;

	while (fgets(line, sizeof(line), f)) {
		if (strstr(line, HAT_CARD_NAME) &&
		    sscanf(line, " %d", &card) == 1)
			break;
		card = -1;
	}
	fclose(f);
	return card;
}

static void print_hat_detected_ok(int card)
{
	printf("OK: ALSA card %d '%s' matches %s driver config.\n",
	       card, HAT_CARD_NAME, HAT_PRODUCT);
	printf("     Driver/overlay check only confirm the mounted board is\n");
	printf("     Waveshare %s (ES7210+ES8156), 3x DIP OFF, then continue.\n",
	       HAT_PRODUCT);
	printf("     playback %s  capture %s\n\n", PLAYBACK_DEV, CAPTURE_DEV);
}

static void print_hat_missing_hint(void)
{
	fprintf(stderr, "FAIL: %s not detected (missing '%s').\n\n",
		HAT_PRODUCT, HAT_CARD_NAME);
	fprintf(stderr, "Please check:\n");
	fprintf(stderr, "  1. HAT on 40-pin header, all 3 DIP switches OFF\n");
	fprintf(stderr, "  2. srpi-config -> Interface Options -> Audio -> "
		"Audio Driver HAT V2\n");
	fprintf(stderr, "  3. Reboot after setup: sync && reboot\n");
	fprintf(stderr, "  4. Run: cat /proc/asound/cards\n");
}

static int check_hat_sound_card(void)
{
	int card = hat_find_card_index();

	if (card < 0) {
		print_hat_missing_hint();
		return -1;
	}
	if (card != HAT_CARD_INDEX) {
		fprintf(stderr,
			"FAIL: %s is card %d, expected card %d for %s / %s\n",
			HAT_PRODUCT, card, HAT_CARD_INDEX,
			PLAYBACK_DEV, CAPTURE_DEV);
		fprintf(stderr, "Run: cat /proc/asound/cards\n");
		return -1;
	}
	print_hat_detected_ok(card);
	return 0;
}

static int open_pcm(snd_pcm_t **pcm, const char *dev, snd_pcm_stream_t stream)
{
	snd_pcm_hw_params_t *hw;
	snd_pcm_uframes_t period = PERIOD_FRAMES;
	snd_pcm_uframes_t buffer = PERIOD_FRAMES * BUFFER_PERIODS;
	unsigned int rate = RATE;
	int ret;

	ret = snd_pcm_open(pcm, dev, stream, 0);
	if (ret < 0) {
		fprintf(stderr, "snd_pcm_open(%s): %s\n", dev, snd_strerror(ret));
		return ret;
	}

	snd_pcm_hw_params_malloc(&hw);
	snd_pcm_hw_params_any(*pcm, hw);
	snd_pcm_hw_params_set_access(*pcm, hw, SND_PCM_ACCESS_RW_INTERLEAVED);
	snd_pcm_hw_params_set_format(*pcm, hw, FORMAT);
	snd_pcm_hw_params_set_channels(*pcm, hw, CHANNELS);
	snd_pcm_hw_params_set_rate_near(*pcm, hw, &rate, NULL);
	snd_pcm_hw_params_set_period_size_near(*pcm, hw, &period, NULL);
	snd_pcm_hw_params_set_buffer_size_near(*pcm, hw, &buffer);
	ret = snd_pcm_hw_params(*pcm, hw);
	snd_pcm_hw_params_free(hw);
	if (ret < 0) {
		fprintf(stderr, "snd_pcm_hw_params: %s\n", snd_strerror(ret));
		snd_pcm_close(*pcm);
		*pcm = NULL;
		return ret;
	}
	return snd_pcm_prepare(*pcm);
}

static void pcm_close(snd_pcm_t **pcm)
{
	if (pcm && *pcm) {
		snd_pcm_close(*pcm);
		*pcm = NULL;
	}
}

static void handle_pcm_error(snd_pcm_t *pcm, snd_pcm_sframes_t ret)
{
	if (ret == -EPIPE) {
		snd_pcm_prepare(pcm);
		return;
	}
	if (ret < 0) {
		fprintf(stderr, "pcm error: %s\n", snd_strerror(ret));
		g_err = 1;
		g_stop = 1;
	}
}

static int pcm_write_frames(snd_pcm_t *pcm, const int16_t *src, size_t frames)
{
	size_t pos = 0;

	while (pos < frames && !g_stop) {
		int16_t chunk[PERIOD_FRAMES * CHANNELS];
		size_t n = PERIOD_FRAMES;

		if (pos + n > frames)
			n = frames - pos;
		memset(chunk, 0, sizeof(chunk));
		if (src)
			memcpy(chunk, src + pos * CHANNELS,
			       n * CHANNELS * sizeof(int16_t));

		snd_pcm_sframes_t w = snd_pcm_writei(pcm, chunk, PERIOD_FRAMES);
		if (w < 0) {
			handle_pcm_error(pcm, w);
			continue;
		}
		pos += n;
	}
	return g_err ? -1 : 0;
}

static int pcm_read_into(snd_pcm_t *pcm, int16_t *dst, size_t cap, size_t *filled)
{
	while (*filled < cap && !g_err && !g_stop) {
		int16_t chunk[PERIOD_FRAMES * CHANNELS];
		snd_pcm_sframes_t n = snd_pcm_readi(pcm, chunk, PERIOD_FRAMES);

		if (n < 0) {
			handle_pcm_error(pcm, n);
			continue;
		}
		if (n == 0)
			continue;

		size_t copy = n;
		if (*filled + copy > cap)
			copy = cap - *filled;
		memcpy(dst + *filled * CHANNELS, chunk,
		       copy * CHANNELS * sizeof(int16_t));
		*filled += copy;
	}
	return g_err ? -1 : 0;
}

static int write_wav(const char *path, const int16_t *data, size_t frames)
{
	FILE *f = fopen(path, "wb");
	uint32_t data_bytes = frames * CHANNELS * sizeof(int16_t);
	uint32_t riff_size = 36 + data_bytes;
	uint16_t audio_format = 1;
	uint16_t num_channels = CHANNELS;
	uint32_t sample_rate = RATE;
	uint32_t byte_rate = sample_rate * num_channels * 2;
	uint16_t block_align = num_channels * 2;
	uint16_t bits_per_sample = 16;
	uint32_t riff_id = 0x46464952;
	uint32_t wave_id = 0x45564157;
	uint32_t fmt_id = 0x20746d66;
	uint32_t fmt_size = 16;
	uint32_t data_id = 0x61746164;

	if (!f)
		return -1;

	fwrite(&riff_id, 4, 1, f);
	fwrite(&riff_size, 4, 1, f);
	fwrite(&wave_id, 4, 1, f);
	fwrite(&fmt_id, 4, 1, f);
	fwrite(&fmt_size, 4, 1, f);
	fwrite(&audio_format, 2, 1, f);
	fwrite(&num_channels, 2, 1, f);
	fwrite(&sample_rate, 4, 1, f);
	fwrite(&byte_rate, 4, 1, f);
	fwrite(&block_align, 2, 1, f);
	fwrite(&bits_per_sample, 2, 1, f);
	fwrite(&data_id, 4, 1, f);
	fwrite(&data_bytes, 4, 1, f);
	fwrite(data, 1, data_bytes, f);
	fclose(f);
	return 0;
}

static int peak_ch(const int16_t *data, size_t frames, int ch)
{
	int peak = 0;

	for (size_t f = 0; f < frames; f++) {
		int v = abs(data[f * CHANNELS + ch]);
		if (v > peak)
			peak = v;
	}
	return peak;
}

static int peak_range(const int16_t *data, size_t frames, int from, int to)
{
	int peak = 0;

	for (int ch = from; ch <= to; ch++) {
		int v = peak_ch(data, frames, ch);
		if (v > peak)
			peak = v;
	}
	return peak;
}

static void print_peaks(const char *tag, const int16_t *data, size_t frames)
{
	printf("%s peak:", tag);
	for (int i = 0; i < CHANNELS; i++)
		printf(" ch%d=%d", i + 1, peak_ch(data, frames, i));
	printf("\n");
}

struct duplex_ctx {
	snd_pcm_t *cap;
	snd_pcm_t *play;
	int16_t *cap_buf;
	size_t cap_frames;
	size_t cap_total;
	const int16_t *voice;
	size_t voice_frames;
	size_t gap_frames;
};

static void *capture_thread(void *arg)
{
	struct duplex_ctx *ctx = arg;

	pcm_read_into(ctx->cap, ctx->cap_buf, ctx->cap_total, &ctx->cap_frames);
	return NULL;
}

static void *playback_thread(void *arg)
{
	struct duplex_ctx *ctx = arg;

	for (int r = 0; r < PLAY_REPEAT && !g_stop; r++) {
		if (pcm_write_frames(ctx->play, ctx->voice, ctx->voice_frames) < 0)
			break;
		if (r < PLAY_REPEAT - 1 &&
		    pcm_write_frames(ctx->play, NULL, ctx->gap_frames) < 0)
			break;
	}
	return NULL;
}

static int run_duplex(struct duplex_ctx *ctx)
{
	pthread_t t_cap, t_play;

	g_stop = 0;
	g_err = 0;
	ctx->cap_frames = 0;

	pthread_create(&t_cap, NULL, capture_thread, ctx);
	usleep(50000);
	pthread_create(&t_play, NULL, playback_thread, ctx);
	pthread_join(t_play, NULL);
	usleep(200000);
	g_stop = 1;
	pthread_join(t_cap, NULL);

	return (g_err || ctx->cap_frames == 0) ? -1 : 0;
}

int main(void)
{
	snd_pcm_t *cap = NULL, *play = NULL;
	int16_t *voice = NULL, *loop = NULL;
	size_t voice_frames = 0;
	size_t gap_frames = (size_t)RATE * GAP_SEC;
	size_t loop_frames;
	struct duplex_ctx ctx;
	int ref_hw, ref_mic;
	int ret = 1;

	if (check_hat_sound_card() < 0)
		return 1;

	printf("format: %dch %dHz 16bit period=%d\n", CHANNELS, RATE, PERIOD_FRAMES);

	printf("=== phase 1: speak into mic (%ds) ===\n", DURATION_SEC);
	if (open_pcm(&cap, CAPTURE_DEV, SND_PCM_STREAM_CAPTURE) < 0)
		return 1;

	voice_frames = 0;
	voice = calloc((size_t)RATE * DURATION_SEC, CHANNELS * sizeof(int16_t));
	if (!voice)
		goto out;

	g_stop = 0;
	g_err = 0;
	if (pcm_read_into(cap, voice, (size_t)RATE * DURATION_SEC, &voice_frames) < 0 ||
	    voice_frames == 0) {
		fprintf(stderr, "FAIL: phase 1 captured no data\n");
		goto out;
	}
	if (write_wav(OUT_RECORD, voice, voice_frames) < 0) {
		fprintf(stderr, "FAIL: cannot write %s\n", OUT_RECORD);
		goto out;
	}
	printf("saved %s (%zu frames, %.1fs)\n", OUT_RECORD, voice_frames,
	       (double)voice_frames / RATE);
	print_peaks("phase1", voice, voice_frames);
	pcm_close(&cap);
	usleep(300000);

	loop_frames = voice_frames * PLAY_REPEAT + gap_frames;
	printf("\n=== phase 2: play voice x%d (gap %ds) + capture ===\n",
	       PLAY_REPEAT, GAP_SEC);
	printf("capture length: %.1fs\n", (double)loop_frames / RATE);

	loop = calloc(loop_frames, CHANNELS * sizeof(int16_t));
	if (!loop)
		goto out;
	if (open_pcm(&play, PLAYBACK_DEV, SND_PCM_STREAM_PLAYBACK) < 0 ||
	    open_pcm(&cap, CAPTURE_DEV, SND_PCM_STREAM_CAPTURE) < 0)
		goto out;

	ctx = (struct duplex_ctx){
		.cap = cap,
		.play = play,
		.cap_buf = loop,
		.cap_total = loop_frames,
		.voice = voice,
		.voice_frames = voice_frames,
		.gap_frames = gap_frames,
	};

	if (run_duplex(&ctx) < 0) {
		fprintf(stderr, "FAIL: phase 2 captured no data\n");
		goto out;
	}
	if (write_wav(OUT_LOOPBACK, loop, ctx.cap_frames) < 0) {
		fprintf(stderr, "FAIL: cannot write %s\n", OUT_LOOPBACK);
		goto out;
	}

	printf("saved %s (%zu frames, %.1fs)\n", OUT_LOOPBACK, ctx.cap_frames,
	       (double)ctx.cap_frames / RATE);
	print_peaks("phase2", loop, ctx.cap_frames);

	ref_hw = peak_range(loop, ctx.cap_frames, 6, 7);
	ref_mic = peak_range(loop, ctx.cap_frames, 0, 3);
	printf("\n");
	if (ref_hw >= PASS_THRESHOLD) {
		printf("PASS: PCB loopback ch7/ch8 (peak %d)\n", ref_hw);
		ret = 0;
		goto out;
	}
	if (ref_mic >= PASS_THRESHOLD) {
		printf("PASS: wired loopback ch1-ch4 (peak %d)\n", ref_mic);
		printf("note: ch7/ch8 peak %d (use ch1-ch4 if speaker taps MIC)\n",
		       ref_hw);
		ret = 0;
		goto out;
	}
	fprintf(stderr, "FAIL: no loopback (ch7/ch8=%d ch1-ch4=%d threshold=%d)\n",
		ref_hw, ref_mic, PASS_THRESHOLD);

out:
	pcm_close(&play);
	pcm_close(&cap);
	free(voice);
	free(loop);
	return ret;
}
