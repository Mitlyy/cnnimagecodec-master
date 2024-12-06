// ImageCodec.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "mcoder.h"

int BitPlaneEncoder(unsigned char *obuffer, unsigned char *layer1, int w1, int h1, int z1, int *bitsize)
{
	BiContextType bin_ctx;
	int code_len = 0;
	EncodingEnvironmentPtr eep;
	unsigned char *bufptr = obuffer;

	eep = arienco_create_encoding_environment();
	arienco_start_encoding(eep, bufptr, &code_len);
	biari_init_context(&bin_ctx, "ctx");

	for (int z = 0;z < z1;z++)
	{
		for (int i = 0;i < h1;i++)
		{
			for (int j = 0;j < w1;j++)
			{
				biari_encode_symbol(eep, layer1[z*h1*w1+i*w1+j], &bin_ctx);
			}
		}
	}

	biari_encode_symbol(eep, 0, &bin_ctx);
	biari_encode_symbol(eep, 1, &bin_ctx);
	biari_encode_symbol(eep, 0, &bin_ctx);
	biari_encode_symbol(eep, 1, &bin_ctx);

	arienco_done_encoding(eep);
	arienco_delete_encoding_environment(eep);
	*bitsize = code_len / 8;

	return code_len / 8;
}

int BitPlaneDecoder(unsigned char *obuffer, unsigned char *layer1, int w1, int h1, int z1, int *offset)
{
	BiContextType bin_ctx;
	int code_len = 0;
	DecodingEnvironmentPtr dep;
	unsigned char *bufptr = obuffer;

	bufptr += (*offset);
	dep = arideco_create_decoding_environment();
	arideco_start_decoding(dep, bufptr, 0, &code_len);
	biari_init_context(&bin_ctx, "ctx");

	for (int z = 0;z < z1;z++)
	{
		for (int i = 0;i < h1;i++)
		{
			for (int j = 0;j < w1;j++)
			{
				layer1[z*h1*w1 + i * w1 + j] = biari_decode_symbol(dep, &bin_ctx);
			}
		}
	}
	*offset = code_len;
	arideco_delete_decoding_environment(dep);

	return 0;
}

int main(int argc, char* argv[])
{
	
	return 0;
}