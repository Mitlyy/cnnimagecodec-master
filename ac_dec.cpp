#include <stdlib.h>
#include <stdio.h>
#include "mcoder.h"
#include <math.h>


/*!
 ************************************************************************
 * \brief
 *    Allocates memory for the DecodingEnvironment struct
 * \return DecodingContextPtr
 *    allocates memory
 ************************************************************************
 */
DecodingEnvironmentPtr arideco_create_decoding_environment()
{
  DecodingEnvironmentPtr dep;

  if ((dep = (DecodingEnvironmentPtr)calloc(1,sizeof(DecodingEnvironment))) == NULL)
    printf("arideco_create_decoding_environment: dep");
  return dep;
}


/*!
 ***********************************************************************
 * \brief
 *    Frees memory of the DecodingEnvironment struct
 ***********************************************************************
 */
void arideco_delete_decoding_environment(DecodingEnvironmentPtr dep)
{
  free(dep);
}


/*!
 ************************************************************************
 * \brief
 *    Initializes the DecodingEnvironment for the arithmetic coder
 ************************************************************************
 */
void arideco_start_decoding(DecodingEnvironmentPtr dep, unsigned char *cpixcode,
                            int firstbyte, int *cpixcode_len )
{
  register unsigned int bbit = 0;
 
  int value = 0;
  dep->Dcodestrm = cpixcode;
  dep->Dcodestrm_len = (unsigned int*)cpixcode_len;
  dep->Dvalue = 0;
  for (int i = 1; i <= BITS_IN_REGISTER; i++)
  {
	Get1Bit(dep->Dcodestrm, *dep->Dcodestrm_len, bbit);
    dep->Dvalue = 2 * dep->Dvalue + bbit;
  }
  dep->Dlow = 0;
  dep->Drange = HALF-2;
}

unsigned int  biari_decode_symbol(DecodingEnvironmentPtr dep, BiContextTypePtr bi_ct)
{
  int bbit;
  register unsigned int low = dep->Dlow;
  register unsigned int value = dep->Dvalue;
  register unsigned int range = dep->Drange;
  unsigned int symbol;

  unsigned int cum = (int)((((long) (value - low) + 1) * bi_ct->freq_all - 1) / range);
  for (int i=0;i<ALPHABET_SIZE;i++)
  {
	if (bi_ct->cum_freq [i] > cum)
	{
		break;		
	}
	symbol = i;
  }

  low = low + (range * bi_ct->cum_freq [symbol]) / bi_ct->freq_all;
  range = (range*bi_ct->freq [symbol])/bi_ct->freq_all;
  range = max(1,range);

  bi_ct->freq[symbol]++;
  bi_ct->freq_all++;
  if (bi_ct->freq_all>16384)
  {
	bi_ct->freq_all=0;
	for (int i=0;i<ALPHABET_SIZE;i++)
	{
		bi_ct->freq[i]=max(1,bi_ct->freq[i]>>1);
		bi_ct->freq_all+=bi_ct->freq[i];
	}
  }
  bi_ct->cum_freq[0]=0;
  for (int i=1;i<=ALPHABET_SIZE;i++)
  {
	bi_ct->cum_freq[i] = bi_ct->cum_freq[i-1]+bi_ct->freq[i-1];
  }

  while (range < QUARTER)
  {
 	if (low >= HALF)
    {
        low -= HALF;
		value -= HALF;
    }
    else 
      if (low < QUARTER)
      {
      }
      else
      {
        low -= QUARTER;
		value -= QUARTER;
      }
    // Double range 
    range <<= 1;
	low <<= 1;
	Get1Bit(dep->Dcodestrm, *dep->Dcodestrm_len, bbit);
    value = (value << 1) | bbit;
  }

  dep->Drange = range;
  dep->Dvalue = value;
  dep->Dlow = low;
 
  return(symbol);
}


